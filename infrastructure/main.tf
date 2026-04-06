# Firm-Pair Merger Prediction — Full-Scale Pipeline Infrastructure
#
# Provisions a single g5.8xlarge EC2 instance for patent vectorization.
# See docs/adr/adr_003_cloud_architecture.md for design rationale.
#
# Usage:
#   cd infrastructure
#   terraform init
#   terraform plan -var="key_name=your-ssh-key" -var="ssh_cidr=YOUR.IP.HERE/32"
#   terraform apply -var="key_name=your-ssh-key" -var="ssh_cidr=YOUR.IP.HERE/32"

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- Variables ---

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "key_name" {
  description = "Name of existing EC2 key pair for SSH access"
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR block for SSH access (e.g., 'YOUR.IP.HERE/32'). Use 'curl ifconfig.me' to find your IP."
  type        = string
}

variable "s3_bucket" {
  description = "Existing S3 bucket for data and results"
  type        = string
  default     = "ubc-torrin"
}

variable "s3_prefix" {
  description = "S3 prefix for this project (namespace isolation)"
  type        = string
  default     = "firm-pair-merger"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.8xlarge"
}

# --- Data Sources ---

# Latest Deep Learning AMI with PyTorch (Ubuntu 22.04)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu 22.04*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Default VPC
data "aws_vpc" "default" {
  default = true
}

# --- Security Group ---

resource "aws_security_group" "pipeline" {
  name        = "firm-pair-merger-pipeline"
  description = "SSH access for patent vectorization pipeline"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "firm-pair-merger-pipeline"
    Project = "firm-pair-merger-prediction"
  }
}

# --- IAM Role for EC2 (S3 access, scoped to project prefix) ---

resource "aws_iam_role" "pipeline" {
  name = "firm-pair-merger-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Project = "firm-pair-merger-prediction"
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name = "firm-pair-merger-s3-access"
  role = aws_iam_role.pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucketUnderPrefix"
        Effect = "Allow"
        Action = "s3:ListBucket"
        Resource = "arn:aws:s3:::${var.s3_bucket}"
        Condition = {
          StringLike = {
            "s3:prefix" = ["${var.s3_prefix}/*"]
          }
        }
      },
      {
        Sid    = "ReadWriteProjectObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "arn:aws:s3:::${var.s3_bucket}/${var.s3_prefix}/*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "pipeline" {
  name = "firm-pair-merger-pipeline-profile"
  role = aws_iam_role.pipeline.name
}

# --- EC2 Instance ---
# g5 instances have NVMe instance store but it must be explicitly mounted.
# We use a large root EBS volume instead for simplicity and persistence.

resource "aws_instance" "pipeline" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.pipeline.id]
  iam_instance_profile   = aws_iam_instance_profile.pipeline.name

  root_block_device {
    volume_size = 200  # GB — enough for data (~5GB), checkpoints (~50GB), model cache
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    s3_bucket = var.s3_bucket
    s3_prefix = var.s3_prefix
  })

  tags = {
    Name    = "firm-pair-merger-pipeline"
    Project = "firm-pair-merger-prediction"
  }
}

# --- Outputs ---

output "instance_id" {
  value = aws_instance.pipeline.id
}

output "public_ip" {
  value = aws_instance.pipeline.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.pipeline.public_ip}"
}

output "ami_name" {
  value = data.aws_ami.deep_learning.name
}

output "estimated_cost_per_hour" {
  value = "~$2.45/hr (g5.8xlarge on-demand, us-west-2)"
}
