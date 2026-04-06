# Firm-Pair Merger Prediction — Full-Scale Pipeline Infrastructure
#
# Provisions a single g5.8xlarge EC2 instance for patent vectorization.
# See docs/adr/adr_003_cloud_architecture.md for design rationale.
#
# Usage:
#   cd infrastructure
#   terraform init
#   terraform plan -var="key_name=dev-environment-key" -var="ssh_cidr=$(curl -s ifconfig.me)/32"
#   terraform apply -var="key_name=dev-environment-key" -var="ssh_cidr=$(curl -s ifconfig.me)/32"
#
# S3 access: After SSH-ing in, run `aws configure --profile torrin` with your
# existing credentials. The bootstrap script pulls data using this profile.

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
  description = "CIDR block for SSH access (e.g., 'YOUR.IP.HERE/32'). Use 'curl -s ifconfig.me' to find your IP."
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

# --- EC2 Instance ---
# No IAM instance profile — S3 access uses Torrin's existing AWS credentials
# configured via `aws configure --profile torrin` after SSH.

resource "aws_instance" "pipeline" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.pipeline.id]

  root_block_device {
    volume_size = 200
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

output "post_ssh_setup" {
  value = "After SSH: aws configure --profile torrin (enter your access key/secret), then run the pipeline"
}
