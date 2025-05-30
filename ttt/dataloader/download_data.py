from huggingface_hub import snapshot_download

# Download to a local directory
snapshot_download(
repo_id="SaylorTwift/the_pile_books3_minus_gutenberg",
local_dir="./pile_books3_data"
)