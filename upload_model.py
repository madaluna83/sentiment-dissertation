from huggingface_hub import HfApi

api = HfApi()

print("Creez repo-ul...")
api.create_repo(
    repo_id="madaluna83/sentiment-distilbert-imdb",
    repo_type="model",
    exist_ok=True
)

print("Urc modelul... (poate dura 2-3 minute)")
api.upload_folder(
    folder_path="D:/sentiment-dissertation/model/model_disertatie_v2",
    repo_id="madaluna83/sentiment-distilbert-imdb",
    repo_type="model"
)

print("Model urcat cu succes!")