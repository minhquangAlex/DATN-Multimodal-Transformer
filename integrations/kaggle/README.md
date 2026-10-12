# Kaggle notebook mirror

Connected notebook: https://www.kaggle.com/code/alexnguyen147/datn-notebook

`datn-notebook/` contains source and metadata downloaded from the existing private notebook. Edit this mirror when updating that combined Kaggle notebook. The separate stage notebooks under `notebooks/` remain independent files.

From the project root in PowerShell:

```powershell
.\scripts\sync_kaggle.ps1 status
.\scripts\sync_kaggle.ps1 pull
.\scripts\sync_kaggle.ps1 save
.\scripts\sync_kaggle.ps1 output
```

`pull` downloads to a new timestamped folder so local edits are preserved. Compare the downloaded notebook against the mirror before replacing it. `save` updates the existing notebook without starting training. `output` retrieves outputs into `results/kaggle-downloads/`.

The notebook's imported `src` is supplied by attached Kaggle datasets. Saving the notebook does not update those datasets or automatically load source from GitHub. Update the source dataset separately when changes in `src/` are needed on Kaggle.

The imported metadata currently disables GPU and attaches multiple source datasets, including legacy ones. Resolve those inputs explicitly before training. Credentials remain outside the project.
