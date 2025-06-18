import time
import runpod
import base64
import io
from PIL import Image
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:7860/sdapi/v1"  # Стандартный порт A1111


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


def change_model(model_name):
    """
    Change the current model checkpoint.
    """
    try:
        payload = {"sd_model_checkpoint": model_name}
        response = automatic_session.post(url=f"{LOCAL_URL}/options", json=payload, timeout=60)

        if response.status_code == 200:
            # Даем время на загрузку модели
            time.sleep(5)
            return True
        else:
            print(f"Failed to change model: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error changing model: {e}")
        return False


def get_available_models():
    """
    Get list of available models.
    """
    try:
        response = automatic_session.get(f"{LOCAL_URL}/sd-models", timeout=30)
        if response.status_code == 200:
            models = response.json()
            return [model["title"] for model in models]
        return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def is_valid_base64_image(b64_string):
    try:
        img_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        return True
    except Exception as e:
        print(f"[INVALID IMAGE] {e}")
        return False


def wait_for_generation_done(timeout=120):  # Увеличен таймаут до 2 минут
    """
    Ждем завершения генерации, проверяя прогресс
    """
    start = time.time()
    print("Waiting for generation to complete...")

    while time.time() - start < timeout:
        try:
            r = automatic_session.get(f"{LOCAL_URL}/progress", timeout=10)
            if r.status_code == 200:
                data = r.json()
                progress = data.get("progress", 0.0)
                state = data.get("state", {})
                job_count = state.get("job_count", 0)

                print(f"Progress: {progress:.2f}, Job count: {job_count}")

                # Генерация завершена когда прогресс 100% И нет активных задач
                if progress >= 1.0 and job_count == 0:
                    print("Generation completed successfully")
                    return True

                # Если есть активные задачи, продолжаем ждать
                if job_count > 0:
                    print(f"Generation in progress... ({job_count} jobs)")

            else:
                print(f"Failed to get progress: {r.status_code}")

        except Exception as e:
            print(f"Error checking progress: {e}")

        time.sleep(2)  # Проверяем каждые 2 секунды

    print(f"Timeout after {timeout} seconds")
    return False


def run_inference(inference_request):
    """
    Run inference on a request.
    """
    # Проверяем, нужно ли сменить модель
    if "model" in inference_request:
        model_name = inference_request.pop("model")  # Убираем из запроса
        print(f"Changing model to: {model_name}")
        if not change_model(model_name):
            return {"error": f"Failed to change model to {model_name}"}

    # Определяем тип запроса
    task_type = inference_request.get("task", "txt2img")

    if task_type == "pipeline":
        endpoint = "txt2img"
    elif task_type == "txt2img":
        endpoint = "txt2img"
    elif task_type == "img2img":
        endpoint = "img2img"
    elif task_type == "get_models":
        return {"models": get_available_models()}
    else:
        endpoint = "txt2img"  # По умолчанию

    if "task" in inference_request:
        inference_request.pop("task")

    if task_type == "pipeline":
        for_reactor_image = inference_request.pop("for_reactor_image")
        if not for_reactor_image:
            return {"error": "for_reactor_image is required for pipeline generation"}

        print("=== PIPELINE GENERATION STARTED ===")
        print("Step 1: txt2img generation...")

        # Убеждаемся что у нас есть все необходимые параметры для txt2img
        txt2img_request = {
            "prompt": inference_request.get("prompt", ""),
            "negative_prompt": inference_request.get("negative_prompt", ""),
            "steps": max(inference_request.get("steps", 20), 15),  # Минимум 15 шагов
            "width": inference_request.get("width", 512),
            "height": inference_request.get("height", 512),
            "cfg_scale": inference_request.get("cfg_scale", 7.0),
            "sampler_name": inference_request.get("sampler_name", "Euler a"),
            "batch_size": 1,
            "n_iter": 1,
            "seed": inference_request.get("seed", -1),
            "restore_faces": False,  # Отключаем restore_faces на первом этапе
            "tiling": False,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
        }

        print(
            f"txt2img parameters: steps={txt2img_request['steps']}, size={txt2img_request['width']}x{txt2img_request['height']}"
        )

        # Этап 1: txt2img генерация
        response = automatic_session.post(url=f"{LOCAL_URL}/{endpoint}", json=txt2img_request, timeout=600)

        if response.status_code != 200:
            return {"error": f"Request failed with status {response.status_code}", "details": response.text}

        response_json = response.json()
        img_b64 = response_json["images"][0]
        print(f"LEN OF GENERATED IMAGE IS {len(img_b64)}")
        print(img_b64)
        if not is_valid_base64_image(img_b64):
            return {"error": "Generated image from txt2img is invalid or corrupted"}
        print("2nd STEP - img2img", flush=True)
        endpoint = "img2img"

        try:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"Image dimensions: {img.size}, mode: {img.mode}")
        except Exception as e:
            print(f"Error analyzing image: {e}")

        if not is_valid_base64_image(img_b64):
            print("ERROR: Generated image from txt2img is invalid!")
            return {"error": "Generated image from txt2img is invalid or corrupted"}

        print("Step 1 completed successfully")

        # Небольшая пауза между этапами
        time.sleep(1)

        print("Step 2: img2img without ReActor (fallback)...")

        # Простая img2img без ReActor как fallback
        second_step_data = {
            "init_images": [img_b64],
            "denoising_strength": 0.3,  # Увеличиваем для лучшего качества
            "prompt": inference_request.get("prompt", ""),
            "negative_prompt": inference_request.get("negative_prompt", ""),
            "steps": max(inference_request.get("steps", 20), 10),  # Минимум 10 шагов для img2img
            "width": inference_request.get("width", 512),
            "height": inference_request.get("height", 512),
            "cfg_scale": inference_request.get("cfg_scale", 7.0),
            "sampler_name": inference_request.get("sampler_name", "Euler a"),
            "batch_size": 1,
            "n_iter": 1,
            "seed": inference_request.get("seed", -1),
            "restore_faces": True,  # Включаем restore_faces на втором этапе
            "do_not_save_samples": True,
            "do_not_save_grid": True,
        }
        response = automatic_session.post(url=f"{LOCAL_URL}/{endpoint}", json=second_step_data, timeout=600)
        # response = automatic_session.post(
        #     url=f'http://127.0.0.1:7860/reactor/image',
        #     json=second_step_data,
        #     timeout=600
        # )
        print("Trying ReActor API directly...")
        second_step_reactor_data = {
            "source_image": "data:imge/png;base64," + for_reactor_image,
            "target_image": "data:image/png;base64," + img_b64,
            "source_faces_index": [0],
            "face_index": [0],
            # "upscaler": "4x_NMKD-Siax_200k",
            "scale": 2,
            "upscale_visibility": 1,
            "face_restorer": "GFPGAN",
            "restorer_visibility": 1,
            "restore_first": 1,
            "model": "inswapper_128.onnx",
            "gender_source": 0,
            "gender_target": 0,
            "save_to_file": 0,
            "result_file_path": "",
            "device": "CUDA",
            "mask_face": 1,
            "select_source": 1,
            "upscale_force": 1,
        }
        response_reactor = automatic_session.post(
            url="http://127.0.0.1:7860/reactor/image", json=second_step_reactor_data, timeout=600
        )
        if response_reactor and "image" in response_reactor:
            print("=== PIPELINE GENERATION COMPLETED WITH REACTOR API ===")
            return {"images": [response_reactor["image"]]}

        print("=== PIPELINE GENERATION COMPLETED ===")
        return response

    print("URL REQUEST")
    print(f"{LOCAL_URL}/{endpoint}")
    try:
        response = automatic_session.post(url=f"{LOCAL_URL}/{endpoint}", json=inference_request, timeout=600)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status {response.status_code}", "details": response.text}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


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
    wait_for_service(url=f"{LOCAL_URL}/sd-models")

    print("WebUI API Service is ready.")
    print("Available models:")
    models = get_available_models()
    for model in models:
        print(f"  - {model}")

    print("Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
