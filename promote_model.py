import mlflow
import dagshub
from mlflow import MlflowClient

dagshub.init(repo_owner='pankajireo74', repo_name='uber-demand-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/pankajireo74/uber-demand-prediction.mlflow")

registered_model_name = 'uber_demand_prediction_model'
staging_stage = "Staging"
production_stage = "Production"

client = MlflowClient()

# List all versions to see what we have
all_versions = client.search_model_versions(f"name='{registered_model_name}'")
print(f"Total model versions found: {len(all_versions)}\n")

for v in all_versions:
    print(f"Version {v.version}: Current stage = {v.current_stage}")

# Get the latest archived version (usually the most recent one)
archived_versions = [v for v in all_versions if v.current_stage == "Archived"]

if not archived_versions:
    print("\n❌ No archived versions to move to Staging!")
    exit(1)

# Get the highest version number from archived
latest_archived = max(archived_versions, key=lambda x: int(x.version))
print(f"\n→ Moving version {latest_archived.version} from Archived → Staging → Production\n")

# Step 1: Move from Archived to Staging
print(f"Step 1: Moving version {latest_archived.version} to Staging...")
client.transition_model_version_stage(
    name=registered_model_name,
    version=latest_archived.version,
    stage=staging_stage,
    archive_existing_versions=False
)
print(f"✓ Version {latest_archived.version} moved to Staging\n")

# Step 2: Move from Staging to Production (this will archive the old production model)
print(f"Step 2: Moving version {latest_archived.version} from Staging to Production...")
model_version_prod = client.transition_model_version_stage(
    name=registered_model_name,
    version=latest_archived.version,
    stage=production_stage,
    archive_existing_versions=True
)

production_version = model_version_prod.version
new_stage = model_version_prod.current_stage

print(f"✓ The model version {production_version} is now in the {new_stage} stage!\n")

# Show final state
print("Final state of all versions:")
all_versions = client.search_model_versions(f"name='{registered_model_name}'")
for v in all_versions:
    print(f"Version {v.version}: {v.current_stage}")