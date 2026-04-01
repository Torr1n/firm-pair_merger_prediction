Hello beautiful people,

To prevent memory overflow, here are the 3 parquet files that (hopefully) is what you need for the embeddings.

1. firm_patents_text_metadata.parquet

What it is: The baseline patents owned by the firms in our Compustat sample (approx. 2.7 million rows).

Columns to use: gvkey, patent_id, title, abstract

2. cited_abstracts.parquet

What it is: The abstracts of every patent cited by our core sample (approx. 3.7 million rows).

Columns to use: patent_id (this is the cited ID), abstract

3. citation_network.parquet

What it is: The mapping file that links the two text datasets together (approx. 46 million edges).

Columns to use: patent_id (the firm's patent), citation_id

Let me know if you have any questions. I cannot wait to get these files off of my laptop.

Best,

Amie
