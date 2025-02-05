from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load OpenAI CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_food(image_path):
    image = Image.open(image_path)
    text_labels = ['Afghani', 'African', 'American', 'Andhra', 'Arabian', 'Argentine', 'Armenian', 'Asian', 'Asian Fusion', 'Assamese', 'Australian', 'Awadhi', 'BBQ', 'Bakery', 'Bar Food', 'Belgian', 'Bengali', 'Beverages', 'Biryani', 'Brazilian', 'Breakfast', 'British', 'Bubble Tea', 'Burger', 'Burmese', 'Cafe', 'Cajun', 'Canadian', 'Cantonese', 'Caribbean', 'Charcoal Grill', 'Chettinad', 'Chinese', 'Coffee and Tea', 'Continental', 'Cuban', 'Curry', 'Deli', 'Desserts', 'Dim Sum', 'Diner', 'Drinks Only', 'European', 'Fast Food', 'Filipino', 'Finger Food', 'Fish and Chips', 'French', 'Fusion', 'German', 'Goan', 'Greek', 'Grill', 'Gujarati', 'Hawaiian', 'Healthy Food', 'Hyderabadi', 'Ice Cream', 'Indian', 'Indonesian', 'International', 'Iranian', 'Irish', 'Italian', 'Japanese', 'Juices', 'Kashmiri', 'Kebab', 'Kerala', 'Korean', 'Latin American', 'Lebanese', 'Lucknowi', 'Maharashtrian', 'Malay', 'Malaysian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Mughlai', 'Nepalese', 'New American', 'North Indian', 'Pakistani', 'Parsi', 'Persian', 'Peruvian', 'Pizza', 'Portuguese', 'Pub Food', 'Rajasthani', 'Ramen', 'Seafood', 'Singaporean', 'South American', 'South Indian', 'Spanish', 'Sri Lankan', 'Steak', 'Street Food', 'Sushi', 'Taiwanese', 'Tapas', 'Tea', 'Tex-Mex', 'Thai', 'Tibetan', 'Turkish', 'Vegetarian', 'Vietnamese']
    
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    top_matches = torch.topk(probs, k=3, dim=1)  # Get top 3 matches
    detected_foods = [text_labels[idx] for idx in top_matches.indices[0].tolist()]
    
    return detected_foods
