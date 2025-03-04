from PIL import Image
import google.generativeai as genai

def detect_food(image_path):
    text_labels = ['Afghani', 'African', 'American', 'Andhra', 'Arabian', 'Argentine', 'Armenian', 'Asian', 'Asian Fusion', 'Assamese', 'Australian', 'Awadhi', 'BBQ', 'Bakery', 'Bar Food', 'Belgian', 'Bengali', 'Beverages', 'Biryani', 'Brazilian', 'Breakfast', 'British', 'Bubble Tea', 'Burger', 'Burmese', 'Cafe', 'Cajun', 'Canadian', 'Cantonese', 'Caribbean', 'Charcoal Grill', 'Chettinad', 'Chinese', 'Coffee and Tea', 'Continental', 'Cuban', 'Curry', 'Deli', 'Desserts', 'Dim Sum', 'Diner', 'Drinks Only', 'European', 'Fast Food', 'Filipino', 'Finger Food', 'Fish and Chips', 'French', 'Fusion', 'German', 'Goan', 'Greek', 'Grill', 'Gujarati', 'Hawaiian', 'Healthy Food', 'Hyderabadi', 'Ice Cream', 'Indian', 'Indonesian', 'International', 'Iranian', 'Irish', 'Italian', 'Japanese', 'Juices', 'Kashmiri', 'Kebab', 'Kerala', 'Korean', 'Latin American', 'Lebanese', 'Lucknowi', 'Maharashtrian', 'Malay', 'Malaysian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Mughlai', 'Nepalese', 'New American', 'North Indian', 'Pakistani', 'Parsi', 'Persian', 'Peruvian', 'Pizza', 'Portuguese', 'Pub Food', 'Rajasthani', 'Ramen', 'Seafood', 'Singaporean', 'South American', 'South Indian', 'Spanish', 'Sri Lankan', 'Steak', 'Street Food', 'Sushi', 'Taiwanese', 'Tapas', 'Tea', 'Tex-Mex', 'Thai', 'Tibetan', 'Turkish', 'Vegetarian', 'Vietnamese']
    
    genai.configure(api_key='AIzaSyA3iXOSQUSF6TUNaaZEXeL10lvyGNKwxOM')
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    prompt = (
        f"Analyze the given image and classify the food into up to three of the following categories: {text_labels}. "
        "Return ONLY a comma-separated list of the top 3 matches, with no extra words or formatting."
    )

    with open(image_path, 'rb') as image_file:
        image_parts = [{
            'mime_type': 'image/jpeg',
            'data': image_file.read()
        }]
    
    response = model.generate_content([prompt, *image_parts])
    
    # Extract response text and format it properly
    if response and response.text:
        top_matches = [match.strip() for match in response.text.split(",")[:3]]
    else:
        top_matches = []

    return top_matches
