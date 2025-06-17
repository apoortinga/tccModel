import ee
import google.auth

def get_ee_credentials():
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine.readonly"
        ]
    )
    return credentials, project

def ee_initialize():
    credentials, project = get_ee_credentials()
    ee.Initialize(
        credentials=credentials,
        project=project,
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
    print(f"Initialized Earth Engine with project: {project}")

# Call it before any EE calls
ee_initialize()

