import base64
import io
from typing import List, Dict, Optional
from PIL import Image
import openai
from config import OPENAI_API_KEY, OPENAI_MODEL

class ChatGPTVisionClient:
    """Client for interacting with ChatGPT Vision API."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_MODEL):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def create_one_shot_prompt(self, reference_images: Dict[int, List[Image.Image]], 
                              test_image: Image.Image) -> List[Dict]:
        """Create one-shot prompting messages with reference images."""
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert ophthalmologist specializing in glaucoma diagnosis from fundus images. 
                You will be provided with reference images showing examples of normal and glaucomatous retinas, 
                followed by a test image that you need to classify."""
            }
        ]
        
        # Add reference images
        content = [
            {
                "type": "text",
                "text": "Here are reference images for glaucoma diagnosis:\n\nNORMAL RETINAS (Label: 0):"
            }
        ]
        
        # Add normal reference images
        for i, img in enumerate(reference_images.get(0, [])):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(img)}"
                }
            })
        
        content.append({
            "type": "text",
            "text": "\nGLAUCOMATOUS RETINAS (Label: 1):"
        })
        
        # Add glaucoma reference images
        for i, img in enumerate(reference_images.get(1, [])):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(img)}"
                }
            })
        
        content.append({
            "type": "text",
            "text": """\n\nNow, please analyze the following TEST IMAGE and classify it as either:
            - 0 (Normal/Negative for glaucoma)
            - 1 (Glaucomatous/Suspicious for glaucoma)
            
            Please provide:
            1. Your classification (0 or 1)
            2. Confidence level (0-100%)
            3. Brief explanation of key features that led to your decision
            
            Format your response as:
            Classification: [0 or 1]
            Confidence: [0-100]%
            Explanation: [Your reasoning]
            
            TEST IMAGE:"""
        })
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{self.encode_image(test_image)}"
            }
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def create_zero_shot_prompt(self, test_image: Image.Image) -> List[Dict]:
        """Create zero-shot prompting messages without reference images."""
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert ophthalmologist specializing in glaucoma diagnosis from fundus images. 
                You will analyze retinal fundus images to detect signs of glaucoma."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Please analyze the following fundus image and classify it as either:
                        - 0 (Normal/Negative for glaucoma)
                        - 1 (Glaucomatous/Suspicious for glaucoma)
                        
                        Look for key glaucoma indicators such as:
                        - Optic disc cupping (increased cup-to-disc ratio)
                        - Neuroretinal rim thinning
                        - Retinal nerve fiber layer defects
                        - Asymmetry between eyes
                        
                        Please provide:
                        1. Your classification (0 or 1)
                        2. Confidence level (0-100%)
                        3. Brief explanation of key features that led to your decision
                        
                        Format your response as:
                        Classification: [0 or 1]
                        Confidence: [0-100]%
                        Explanation: [Your reasoning]"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.encode_image(test_image)}"
                        }
                    }
                ]
            }
        ]
        
        return messages
    
    def get_prediction(self, messages: List[Dict]) -> Optional[Dict]:
        """Get prediction from ChatGPT."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent medical predictions
            )
            
            content = response.choices[0].message.content
            return self.parse_response(content)
            
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return None
    
    def parse_response(self, response_text: str) -> Dict:
        """Parse ChatGPT response to extract classification, confidence, and explanation."""
        result = {
            'classification': None,
            'confidence': None,
            'explanation': '',
            'raw_response': response_text
        }
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Classification:'):
                try:
                    result['classification'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Confidence:'):
                try:
                    conf_str = line.split(':')[1].strip().replace('%', '')
                    result['confidence'] = float(conf_str)
                except:
                    pass
            elif line.startswith('Explanation:'):
                result['explanation'] = line.split(':', 1)[1].strip()
        
        # If parsing failed, try to extract classification from the response
        if result['classification'] is None:
            if '1' in response_text and 'glaucoma' in response_text.lower():
                result['classification'] = 1
            elif '0' in response_text and ('normal' in response_text.lower() or 'negative' in response_text.lower()):
                result['classification'] = 0
        
        return result
