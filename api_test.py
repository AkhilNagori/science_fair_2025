import requests

def ocr_space_file(filename, api_key='helloworld', language='eng', ocr_engine=1):
    # Define the API endpoint and set up the payload for the request
    api_url = 'https://api.ocr.space/parse/image'
    payload = {
        'isOverlayRequired': False,
        'apikey': api_key,
        'language': language,
        'OCREngine': ocr_engine,
    }
    
    # Open the image file in binary mode and send the request
    with open(filename, 'rb') as f:
        response = requests.post(api_url, files={'filename': f}, data=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        return result['ParsedResults'][0]['ParsedText'] if 'ParsedResults' in result else None
    else:
        print(f"Error: {response.status_code}")
        return None

# Replace 'your_image.png' with your actual image file path
text = ocr_space_file(filename='test_image_travis.png', api_key='helloworld')
print("Extracted Text:", text)
