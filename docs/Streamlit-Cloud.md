# Streamlit Cloud deployment

RetailPulse is prepared for free-tier Streamlit Cloud deployment using the dashboard entrypoint at `retailpulse_dashboard.py`.

## What to deploy

Deploy only the Streamlit dashboard app. Do not deploy the Docker or Kubernetes infrastructure to Streamlit Cloud.

## Checklist

 - `requirements-streamlit.txt` includes Streamlit runtime dependencies (lightweight, excludes heavy ML libs).
- `retailpulse_dashboard.py` is the main entrypoint.
- Artifact files under `processed/` are present in the repository or generated before deployment.
- Heavy local-only infrastructure code is isolated from the dashboard entrypoint.

## Deployment steps

1. Push the repository to GitHub.
2. Open Streamlit Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to `retailpulse_dashboard.py`.
5. In Streamlit Cloud, set the requirements file to `requirements-streamlit.txt` (or paste dependencies).
6. Deploy.

## Notes

- Keep the dashboard artifact-driven so it can render without training jobs.
- If a required artifact is missing, the app should show a user-friendly warning instead of crashing.
- Use free-tier-friendly package sizes and avoid unnecessary runtime dependencies.
