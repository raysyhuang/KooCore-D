# KooCore-D Dashboard

Read-only observability dashboard for KooCore-D stock picking system.

## Architecture (Option A: GitHub Artifact Pull)

```
KooCore-D (private, engine)
  └─ GitHub Actions (daily scan)
        └─ uploads outputs as artifact
             ↓
KooCore-D-Dashboard (public)
  └─ pulls artifact via GitHub API
        └─ Heroku auto-deploy
             └─ Streamlit + Plotly dashboard
```

**Key principle**: Dashboard never runs scans. It only reads versioned artifacts.

## Features

- **Overview**: Pick counts, hit rates, summary metrics
- **Phase-5 Learning**: Hit rate by regime, source, rank decay analysis
- **Picks Explorer**: Browse daily picks by date
- **Scorecards**: View Phase-5 analysis results
- **Raw Files**: Browse all files in the artifact

## Local Development

```bash
# Set environment variables
export GITHUB_REPO="raysyhuang/KooCore-D"
export GITHUB_TOKEN="your_github_token"
export ARTIFACT_NAME="koocore-outputs"

# Install and run
pip install -r requirements.txt
streamlit run app.py
```

## Heroku Deployment

### 1. Create Heroku App

```bash
heroku create koocore-dashboard
```

### 2. Set Config Vars

```bash
heroku config:set GITHUB_REPO=raysyhuang/KooCore-D
heroku config:set ARTIFACT_NAME=koocore-outputs
heroku config:set GITHUB_TOKEN=your_token_here
```

### 3. Deploy

```bash
git push heroku main
```

Or connect to GitHub for auto-deploy.

## GitHub Token Setup

Create a fine-grained token at https://github.com/settings/tokens?type=beta

**Required permissions:**
- Repository access: Only `KooCore-D`
- Permissions:
  - Actions: Read
  - Contents: Read (optional but recommended)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_REPO` | Yes | Source repo, e.g. `raysyhuang/KooCore-D` |
| `GITHUB_TOKEN` | Yes | GitHub token with Actions read |
| `ARTIFACT_NAME` | No | Artifact name (default: `koocore-outputs`) |
| `GITHUB_BRANCH` | No | Filter by branch (optional) |

## Safety Rules

- ❌ No writes to source repo
- ❌ No model logic
- ❌ No scan execution
- ✅ Read-only artifact consumption
- ✅ Pure visualization

## Data Flow

1. KooCore-D runs daily scans via GitHub Actions
2. Workflow uploads `outputs/` as artifact named `koocore-outputs`
3. Dashboard fetches latest artifact via GitHub API
4. Streamlit renders charts from artifact data

## File Structure

```
KooCore-D-Dashboard/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku process file
└── README.md          # This file
```
