import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:7860"  # Стандартный порт A1111
API_URL = f"{LOCAL_URL}/sdapi/v1"


automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount("http://", HTTPAdapter(max_retries=retries))

# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #


def wait_for_service(url):
    """
    Check if the service is ready to receive requests.
    """
    retries = 0
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            retries += 1
            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)


def get_available_models():
    """
    Get list of available models.
    """
    try:
        response = automatic_session.get(f"{API_URL}/sd-models", timeout=30)
        if response.status_code == 200:
            models = response.json()
            return [model["title"] for model in models]
        return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def run_inference(inference_request: dict):
    """
    Run the inference.
    """
    source_image = inference_request.get("source_image")
    target_image = inference_request.get("target_image")
    body = {
        "source_image": source_image,
        "target_image": target_image,
        "resize_mode": "Just resize",
        "sampling_method": "DPM++ 2M",
        "schedule_type": "Karras",
        "sampling_steps": 20,
        "width": 768,
        "height": 1152,
        "batch_count": 1,
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0,
        "model": "inswapper_128.onnx",
        "device": "CPU",
        "det_thresh": 0.5,
        "det_maxnum": 0,
    }
    result = automatic_session.post(f"{LOCAL_URL}/reactor/image", json=body)
    return result.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #


def handler(event):
    """
    This is the handler function that will be called by the serverless.
    """
    try:
        result = run_inference(event["input"])
        return result
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}


if __name__ == "__main__":
    print("Waiting for WebUI API Service...")
    wait_for_service(url=f"{API_URL}/sd-models")

    print("WebUI API Service is ready.")
    print("Available models:")
    models = get_available_models()
    for model in models:
        print(f"  - {model}")

    print("Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
