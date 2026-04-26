
import gradio as gr
from fastai.vision.all import *
import torch
import torchvision.transforms as T

learn = load_learner('creature_model.pkl')
model = learn.model
model.eval()

creature_info = {
    'ant': {
        'danger': '🟡 Low Danger',
        'useful': '✅ Useful — aerates soil and controls other pests',
        'fact': 'Ants can carry 50 times their own body weight'
    },
    'aphids': {
        'danger': '🟠 Medium Danger to crops',
        'useful': '❌ Harmful — damages crops by sucking plant sap',
        'fact': 'A single aphid can produce 80 offspring in one week'
    },
    'armyworm': {
        'danger': '🔴 High Danger to crops',
        'useful': '❌ Harmful — destroys entire fields of crops overnight',
        'fact': 'Armyworms travel in large groups like an army — hence the name'
    },
    'beetle': {
        'danger': '🟡 Low Danger to humans',
        'useful': '✅ Mostly useful — many beetles decompose dead matter',
        'fact': 'Beetles make up 25% of all known animal species on earth'
    },
    'grasshopper': {
        'danger': '🟠 Medium Danger to crops',
        'useful': '❌ Harmful in large numbers — destroys agricultural fields',
        'fact': 'Grasshoppers can jump 20 times their own body length'
    },
    'lizard': {
        'danger': '🟢 No Danger to humans',
        'useful': '✅ Very useful — eats mosquitoes, cockroaches and harmful insects',
        'fact': 'A single lizard can eat hundreds of insects every day'
    },
    'mosquito': {
        'danger': '🔴 High Danger — spreads Malaria, Dengue, Chikungunya',
        'useful': '❌ Harmful — most dangerous creature to humans worldwide',
        'fact': 'Mosquitoes kill more humans per year than any other animal'
    },
    'spider': {
        'danger': '🟡 Low Danger — most house spiders are harmless',
        'useful': '✅ Useful — controls mosquito and fly populations',
        'fact': 'Spiders eat more insects per year than the total weight of all humans'
    }
}

tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def identify_creature(img):
    tensor = tfms(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        preds = torch.softmax(model(tensor), dim=1)[0]
    
    classes = ['ant', 'aphids', 'armyworm', 'beetle', 
               'grasshopper', 'lizard', 'mosquito', 'spider']
    
    top_idx = preds.argmax().item()
    top_class = classes[top_idx]
    confidence = float(preds[top_idx]) * 100
    info = creature_info[top_class]
    
    result = f"""
## {top_class.upper()} — {confidence:.1f}% confident

**{info['danger']}**

**Impact:** {info['useful']}

**Interesting Fact:** {info['fact']}

---
*Identified using AI trained on insects common in Andhra Pradesh*
    """
    return result

app = gr.Interface(
    fn=identify_creature,
    inputs=gr.Image(type='pil'),
    outputs=gr.Markdown(),
    title="🦟 Creature Identifier — Andhra Pradesh Edition",
    description="Upload a photo of any creature found in your campus or hostel. Get instant identification, danger level, and interesting facts."
)

app.launch(share=True)
