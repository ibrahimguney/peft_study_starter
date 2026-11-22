# Sync script for local Windows OneDrive repository
# Target repo: https://github.com/ibrahimguney/peft_study_starter
# Local path: C:\Users\ibrah\OneDrive - IZU\Github\peft_study_starter

$RepoPath = "C:\Users\ibrah\OneDrive - IZU\Github\peft_study_starter"
Set-Location $RepoPath
git status
git add .
git commit -m "Add ablation config, results analysis script, and manuscript draft outline"
git push origin main
