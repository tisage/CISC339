import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import ast
import os
import glob
import json

def get_device():
    """Select the best available hardware acceleration device."""
    if torch.backends.mps.is_available():
        print("Hardware Match: Apple Silicon (MPS) acceleration enabled!")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Hardware Match: CUDA acceleration enabled!")
        return torch.device("cuda")
    else:
        print("Hardware Match: Defaulting to CPU.")
        return torch.device("cpu")

def main():
    device = get_device()
    
    # 1. Load Donut Processor and Model
    # This model directly converts receipt/document images into structured JSON.
    model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    print(f"Loading '{model_id}' from HuggingFace...")
    print("Initial run may download model weights (~800MB). Please wait.")
    
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    
    # Move model to acceleration hardware
    model.to(device)
    model.eval()

    # 2. Search for images to process (Supports multiple, e.g., sample_1.png, sample_2.jpg, etc.)
    search_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = []
    
    # Search for all common image file extensions in the directory
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(search_dir, ext)))
        
    if not image_paths:
        print(f"No images found in {search_dir}!")
        print("Demo Mode: Automatically generating 'sample_document.jpg' to test the pipeline...")
        demo_image = os.path.join(search_dir, "sample_document.jpg")
        Image.new('RGB', (800, 600), color='white').save(demo_image)
        image_paths = [demo_image]
        
    print(f"\nFound {len(image_paths)} image(s) ready for document analysis.")
    
    # Prepare specific prompt required by the model
    task_prompt = "<s_cord-v2>" # Special token for cord-v2 dataset
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 3. Process each image in a loop
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        print("\n" + "="*50)
        print(f"Analyzing image: [{file_name}] ...")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping image. File may be corrupted or unreadable: {e}")
            continue

        # Image Preprocessing
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Run model inference to generate structured document (JSON format)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Post-processing and Decoding
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        # Extract structured Python dictionary (JSON data structure)
        organized_data = processor.token2json(sequence)
        
        # 4. Generate a JSON file with the same base name
        # Example: sample_1.png -> sample_1.json
        base_name, _ = os.path.splitext(image_path)
        output_file_path = f"{base_name}.json"
        
        # Write to file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Processing complete! Extracted features saved to:\n---> {output_file_path}")
        print("Preview of the generated content:")
        print(json.dumps(organized_data, indent=4, ensure_ascii=False))

    print("\nAll document images processed successfully!")

if __name__ == "__main__":
    main()
