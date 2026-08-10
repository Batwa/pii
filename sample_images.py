"""
Create sample images with text PII for testing
(Since we can't create faces programmatically, we'll create text-based PII)
"""
from PIL import Image, ImageDraw, ImageFont
import os
from config import DATA_INPUT

def create_sample_images():
    """Create sample images with PII text"""
    # Create images directory
    image_dir = os.path.join(DATA_INPUT, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Sample PII data to put in images
    sample_data = [
        {
            'filename': 'business_card.png',
            'text_items': [
                ('John Smith', (50, 50)),
                ('CEO, Tech Company', (50, 80)),
                ('john.smith@company.com', (50, 110)),
                ('Phone: 555-123-4567', (50, 140)),
                ('123 Business Ave, Suite 100', (50, 170)),
                ('New York, NY 10001', (50, 200))
            ],
            'size': (400, 300),
            'bg_color': (255, 255, 255)
        },
        {
            'filename': 'employee_badge.png',
            'text_items': [
                ('EMPLOYEE ID: E12345', (30, 30)),
                ('Sarah Johnson', (30, 70)),
                ('Engineering Department', (30, 100)),
                ('s.johnson@company.com', (30, 130)),
                ('Emergency: 555-987-6543', (30, 160)),
                ('SSN: 123-45-6789', (30, 190))
            ],
            'size': (350, 250),
            'bg_color': (240, 240, 255)
        },
        {
            'filename': 'receipt.png',
            'text_items': [
                ('RECEIPT', (100, 20)),
                ('Date: 2024-01-15', (20, 60)),
                ('Customer: Mike Wilson', (20, 90)),
                ('Phone: (555) 444-3333', (20, 120)),
                ('Email: mike.w@gmail.com', (20, 150)),
                ('Credit Card: **** **** **** 1234', (20, 180)),
                ('Total: $49.99', (20, 210)),
                ('Thank you for your business!', (20, 250))
            ],
            'size': (300, 300),
            'bg_color': (255, 255, 240)
        },
        {
            'filename': 'form.png',
            'text_items': [
                ('APPLICATION FORM', (80, 20)),
                ('Name: Lisa Brown', (20, 60)),
                ('DOB: 03/15/1985', (20, 90)),
                ('Address: 789 Oak Street', (20, 120)),
                ('City: Los Angeles, CA 90210', (20, 150)),
                ('Phone: +1-555-777-8888', (20, 180)),
                ('Email: lisa.brown@personal.net', (20, 210)),
                ('Signature: _________________', (20, 250))
            ],
            'size': (350, 300),
            'bg_color': (255, 250, 240)
        }
    ]
    
    created_files = []
    
    for sample in sample_data:
        # Create image
        image = Image.new('RGB', sample['size'], sample['bg_color'])
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)  # macOS
            except:
                font = ImageFont.load_default()
        
        # Draw text items
        for text, position in sample['text_items']:
            draw.text(position, text, fill=(0, 0, 0), font=font)
        
        # Save image
        filepath = os.path.join(image_dir, sample['filename'])
        image.save(filepath)
        created_files.append(filepath)
        print(f"✅ Created: {sample['filename']}")
    
    print(f"\n🎉 Created {len(created_files)} sample images in {image_dir}")
    print("\n📋 Sample images contain:")
    print("  - Names, emails, phone numbers")
    print("  - Addresses and SSNs")  
    print("  - Credit card info")
    print("  - Business information")
    
    return created_files

def create_instruction_image():
    """Create an instruction image for users who want to test with faces"""
    image_dir = os.path.join(DATA_INPUT, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Create instruction image
    image = Image.new('RGB', (500, 400), (240, 248, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    instructions = [
        ("FACE DETECTION TEST", (150, 30), font_large),
        ("To test face detection:", (50, 80), font_small),
        ("1. Add photos with faces to this folder:", (50, 110), font_small),
        ("   data/input/images/", (50, 130), font_small),
        ("2. Run: python image_detector.py", (50, 160), font_small),
        ("3. Check results in data/output/images/", (50, 190), font_small),
        ("", (50, 220), font_small),
        ("Supported formats:", (50, 250), font_small),
        (".jpg, .jpeg, .png, .bmp, .tiff", (50, 270), font_small),
        ("", (50, 300), font_small),
        ("Current sample images test TEXT PII only", (50, 330), font_small),
        ("(names, emails, phones, addresses, SSNs)", (50, 350), font_small)
    ]
    
    for text, position, font in instructions:
        draw.text(position, text, fill=(0, 0, 0), font=font)
    
    filepath = os.path.join(image_dir, "README_face_testing.png")
    image.save(filepath)
    print(f"📋 Created instruction image: README_face_testing.png")

if __name__ == "__main__":
    print("🖼️ Creating sample images with PII...")
    create_sample_images()
    create_instruction_image()
    print("\n🚀 Ready to test image PII detection!")
    print("   Run: python image_detector.py")
