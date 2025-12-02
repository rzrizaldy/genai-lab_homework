"""
Seller Quality Optimizer - Agentic Workflow Engine
"""

import os
import json
import random
import requests
import base64
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import pandas as pd

class WorkflowEngine:
    def __init__(self, api_key: str, data_dir: str, output_dir: str):
        self.client = OpenAI(api_key=api_key)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.products_df = pd.read_csv(self.data_dir / "products.csv")
        self.reviews_df = pd.read_csv(self.data_dir / "reviews.csv")
        
        # Workflow state
        self.current_step = 0
        self.steps = []
        self.result = {}
    
    def log_step(self, name: str, status: str, data: dict = None):
        step = {
            "step": len(self.steps) + 1,
            "name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.steps.append(step)
        return step
    
    def select_random_product_with_reviews(self, search_query: str = "", max_attempts: int = 50) -> dict:
        """Step 1: Select a random product that has reviews, optionally filtered by search query"""
        self.log_step("Selecting Random Product", "running")
        
        # Filter products by search query if provided
        if search_query:
            # Case-insensitive search in title and brand
            mask = (
                self.products_df['title'].str.lower().str.contains(search_query.lower(), na=False) |
                self.products_df['brand_name'].str.lower().str.contains(search_query.lower(), na=False)
            )
            filtered_products = self.products_df[mask]
            
            if len(filtered_products) == 0:
                raise Exception(f"No products found matching '{search_query}'")
            
            search_pool = filtered_products
        else:
            search_pool = self.products_df
        
        for attempt in range(max_attempts):
            # Random product from pool
            idx = random.randint(0, len(search_pool) - 1)
            product = search_pool.iloc[idx]
            asin = product['asin']
            
            # Check for reviews
            reviews = self.reviews_df[self.reviews_df['productASIN'] == asin]
            
            if len(reviews) > 0:
                self.result['product'] = {
                    'asin': asin,
                    'title': product['title'],
                    'description': product.get('about_item', ''),
                    'brand': product.get('brand_name', ''),
                    'rating': product.get('rating_stars', ''),
                    'image_url': self._extract_first_image(product.get('all_images', ''))
                }
                self.result['reviews'] = reviews.to_dict('records')
                self.result['attempt'] = attempt + 1
                self.result['search_query'] = search_query
                self.result['matches_found'] = len(search_pool) if search_query else len(self.products_df)
                
                self.log_step("Selecting Random Product", "completed", {
                    "asin": asin,
                    "title": product['title'][:50] + "...",
                    "review_count": len(reviews),
                    "attempts": attempt + 1,
                    "search_matches": len(search_pool) if search_query else "all"
                })
                return self.result['product']
        
        raise Exception(f"Could not find product with reviews after {max_attempts} attempts")
    
    def _extract_first_image(self, images_str: str) -> str:
        """Extract first image URL from the images list string"""
        try:
            if pd.isna(images_str):
                return ""
            images = eval(images_str)
            return images[0] if images else ""
        except:
            return ""
    
    def analyze_sentiment(self) -> dict:
        """Step 2: Analyze sentiment of reviews"""
        self.log_step("Analyzing Sentiment", "running")
        
        reviews = self.result.get('reviews', [])
        product = self.result.get('product', {})
        
        sentiments = []
        discrepancies = []
        
        for review in reviews[:10]:  # Limit to 10 reviews for speed
            review_text = review.get('reviewText', '') or review.get('cleaned_review_text', '')
            rating = review.get('rating', 3)
            
            if not review_text or len(review_text) < 10:
                continue
            
            prompt = f"""Analyze this product review for sentiment and visual/quality discrepancies.

Product: {product.get('title', 'Unknown')}
Description: {product.get('description', '')[:500]}

Review ({rating} stars):
"{review_text[:1000]}"

Respond with ONLY valid JSON:
{{
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "sentiment_score": number between -1.0 and 1.0,
  "emotional_tone": "string",
  "visual_discrepancy_found": boolean,
  "discrepancy_details": "string or None"
}}"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
                content = response.choices[0].message.content
                content = content.replace("```json", "").replace("```", "").strip()
                analysis = json.loads(content)
                sentiments.append(analysis)
                
                if analysis.get('visual_discrepancy_found') and analysis.get('discrepancy_details'):
                    discrepancies.append(analysis['discrepancy_details'])
            except Exception as e:
                print(f"Error analyzing review: {e}")
                continue
        
        # Aggregate
        if sentiments:
            avg_score = sum(s.get('sentiment_score', 0) for s in sentiments) / len(sentiments)
            overall = "positive" if avg_score > 0.3 else "negative" if avg_score < -0.3 else "mixed"
        else:
            avg_score = 0
            overall = "unknown"
        
        self.result['sentiment_analysis'] = {
            'overall_sentiment': overall,
            'avg_score': round(avg_score, 2),
            'review_count_analyzed': len(sentiments),
            'discrepancies': discrepancies[:5],  # Top 5
            'individual_sentiments': sentiments
        }
        
        self.log_step("Analyzing Sentiment", "completed", {
            "overall": overall,
            "avg_score": round(avg_score, 2),
            "discrepancy_count": len(discrepancies)
        })
        
        return self.result['sentiment_analysis']
    
    def generate_dalle_prompt(self) -> str:
        """Step 3: Generate a DALL-E prompt based on complaints"""
        self.log_step("Generating Image Prompt", "running")
        
        product = self.result.get('product', {})
        sentiment = self.result.get('sentiment_analysis', {})
        discrepancies = sentiment.get('discrepancies', [])
        
        prompt = f"""Create a DALL-E prompt to visualize a professional product photo that subtly shows the issues customers complained about.

Product: {product.get('title', 'Unknown')}
Customer Complaints: {', '.join(discrepancies) if discrepancies else 'General quality issues'}

REQUIREMENTS:
- Professional product photography style (like Amazon/e-commerce listing)
- Clean white or gradient studio background
- Professional lighting setup
- NO human models or people - product only
- Show the product from an angle that reveals the quality issues mentioned
- Photorealistic, high-end commercial photography aesthetic

Respond with ONLY valid JSON:
{{
  "dalle_prompt": "Professional e-commerce product photography of... (max 400 chars)"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            dalle_prompt = result.get('dalle_prompt', '')
        except Exception as e:
            dalle_prompt = f"Professional product photography of {product.get('title', 'product')}. Show realistic version with quality issues. White background, studio lighting."
        
        self.result['dalle_prompt'] = dalle_prompt
        
        self.log_step("Generating Image Prompt", "completed", {
            "prompt_length": len(dalle_prompt)
        })
        
        return dalle_prompt
    
    def generate_images(self) -> list:
        """Step 4: Generate 2 images using DALL-E 3 and GPT-Image-1"""
        self.log_step("Generating Images", "running")
        
        dalle_prompt = self.result.get('dalle_prompt', 'Product photo')
        product = self.result.get('product', {})
        asin = product.get('asin', 'unknown')
        
        images = []
        
        # Image 1: DALL-E 3
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                size="1024x1024",
                quality="hd",
                style="natural",
                n=1
            )
            url = response.data[0].url
            
            # Download and save
            img_data = requests.get(url).content
            filename = f"{asin}_dalle3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            images.append({
                'model': 'DALL-E 3',
                'url': url,
                'local_path': str(filepath),
                'filename': filename
            })
        except Exception as e:
            print(f"Error generating DALL-E 3 image: {e}")
            images.append({'model': 'DALL-E 3', 'error': str(e)})
        
        # Image 2: DALL-E 2
        try:
            response = self.client.images.generate(
                model="dall-e-2",
                prompt=dalle_prompt[:1000],  # DALL-E 2 has shorter prompt limit
                size="1024x1024",
                n=1
            )
            url = response.data[0].url
            
            # Download and save
            img_data = requests.get(url).content
            filename = f"{asin}_dalle2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            images.append({
                'model': 'DALL-E 2',
                'url': url,
                'local_path': str(filepath),
                'filename': filename
            })
        except Exception as e:
            print(f"Error generating DALL-E 2 image: {e}")
            images.append({'model': 'DALL-E 2', 'error': str(e)})
        
        self.result['generated_images'] = images
        
        self.log_step("Generating Images", "completed", {
            "images_generated": len([i for i in images if 'filename' in i])
        })
        
        return images
    
    def analyze_images(self) -> dict:
        """Step 5: Use GPT-4o Vision to analyze and compare images"""
        self.log_step("Analyzing Images with Vision", "running")
        
        images = self.result.get('generated_images', [])
        sentiment = self.result.get('sentiment_analysis', {})
        discrepancies = sentiment.get('discrepancies', [])
        
        valid_images = [img for img in images if 'filename' in img]
        
        if len(valid_images) < 2:
            self.log_step("Analyzing Images with Vision", "completed", {"error": "Not enough images"})
            self.result['vision_analysis'] = {"error": "Not enough images to compare"}
            return self.result['vision_analysis']
        
        # Build vision message with base64 encoded local images
        content = [
            {
                "type": "text",
                "text": f"""Compare these 2 AI-generated product images against customer complaints.

Image 1: {valid_images[0].get('model', 'Model 1')}
Image 2: {valid_images[1].get('model', 'Model 2')}

Complaints: {', '.join(discrepancies) if discrepancies else 'General quality issues'}

Score each image (1-10) on:
- Professional product photography quality
- How well it depicts the described quality issues
- Commercial/e-commerce suitability

Output JSON:
{{
  "image1_score": number,
  "image1_notes": "string",
  "image2_score": number,
  "image2_notes": "string",
  "best_image": 1 or 2,
  "summary": "string explaining which is better for seller improvement"
}}"""
            }
        ]
        
        # Add images as base64
        for img in valid_images[:2]:
            local_path = img.get('local_path')
            if local_path and Path(local_path).exists():
                with open(local_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
            elif img.get('url') and not img['url'].startswith('local:'):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img['url']}
                })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(result_text)
        except Exception as e:
            analysis = {
                "error": str(e),
                "image1_score": 5,
                "image2_score": 5,
                "best_image": 1,
                "summary": "Unable to analyze images"
            }
        
        self.result['vision_analysis'] = analysis
        
        self.log_step("Analyzing Images with Vision", "completed", {
            "best_image": analysis.get('best_image'),
            "scores": f"{analysis.get('image1_score', 'N/A')} vs {analysis.get('image2_score', 'N/A')}"
        })
        
        return analysis
    
    def save_results(self) -> str:
        """Step 6: Save all results to output folder (JSON + CSV)"""
        self.log_step("Saving Results", "running")
        
        product = self.result.get('product', {})
        asin = product.get('asin', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sentiment = self.result.get('sentiment_analysis', {})
        vision = self.result.get('vision_analysis', {})
        images = self.result.get('generated_images', [])
        
        # Create analysis JSON
        output = {
            "asin": asin,
            "product_title": product.get('title', ''),
            "sentiment": sentiment,
            "dalle_prompt": self.result.get('dalle_prompt', ''),
            "images": images,
            "vision_analysis": vision,
            "workflow_steps": self.steps,
            "analyzed_at": datetime.now().isoformat()
        }
        
        json_filename = f"analysis_{asin}_{timestamp}.json"
        json_filepath = self.output_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save to CSV (append mode)
        csv_filepath = self.output_dir / "analysis_results.csv"
        csv_row = {
            "asin": asin,
            "product_title": product.get('title', ''),
            "brand": product.get('brand', ''),
            "overall_sentiment": sentiment.get('overall_sentiment', ''),
            "avg_sentiment_score": sentiment.get('avg_score', ''),
            "reviews_analyzed": sentiment.get('review_count_analyzed', 0),
            "discrepancies": ' | '.join(sentiment.get('discrepancies', [])),
            "dalle_prompt": self.result.get('dalle_prompt', ''),
            "image1_model": images[0].get('model', '') if len(images) > 0 else '',
            "image1_filename": images[0].get('filename', '') if len(images) > 0 else '',
            "image1_score": vision.get('image1_score', ''),
            "image2_model": images[1].get('model', '') if len(images) > 1 else '',
            "image2_filename": images[1].get('filename', '') if len(images) > 1 else '',
            "image2_score": vision.get('image2_score', ''),
            "best_image": vision.get('best_image', ''),
            "vision_summary": vision.get('summary', ''),
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Check if CSV exists to determine if we need headers
        write_header = not csv_filepath.exists()
        csv_df = pd.DataFrame([csv_row])
        csv_df.to_csv(csv_filepath, mode='a', header=write_header, index=False)
        
        self.result['output_file'] = str(json_filepath)
        self.result['csv_file'] = str(csv_filepath)
        
        self.log_step("Saving Results", "completed", {
            "json_file": json_filename,
            "csv_file": "analysis_results.csv"
        })
        
        return str(json_filepath)
    
    def _clean_result_for_json(self, obj):
        """Clean result object for JSON serialization (handle NaN, etc.)"""
        import math
        if isinstance(obj, dict):
            return {k: self._clean_result_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_result_for_json(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif pd.isna(obj):
            return None
        return obj
    
    def run_full_workflow(self) -> dict:
        """Execute the complete workflow"""
        try:
            self.select_random_product_with_reviews()
            self.analyze_sentiment()
            self.generate_dalle_prompt()
            self.generate_images()
            self.analyze_images()
            self.save_results()
            
            # Clean result for JSON serialization
            clean_result = self._clean_result_for_json(self.result)
            
            return {
                "success": True,
                "result": clean_result,
                "steps": self.steps
            }
        except Exception as e:
            self.log_step("Workflow Error", "failed", {"error": str(e)})
            return {
                "success": False,
                "error": str(e),
                "steps": self.steps
            }
    
    def run_full_workflow_with_progress(self, search_query: str = ""):
        """Execute the complete workflow with progress updates (generator)"""
        try:
            # Step 1: Select Product
            if search_query:
                yield {"type": "step", "step": 1, "status": "running", "message": f"Searching for products matching '{search_query}'..."}
            else:
                yield {"type": "step", "step": 1, "status": "running", "message": "Selecting random product with reviews..."}
            
            product = self.select_random_product_with_reviews(search_query=search_query)
            
            matches_info = f" (from {self.result.get('matches_found', '?')} matches)" if search_query else ""
            yield {
                "type": "step", 
                "step": 1, 
                "status": "completed", 
                "message": f"Selected: {product['title'][:50]}...{matches_info}",
                "data": {
                    "asin": product['asin'],
                    "title": product['title'][:60],
                    "brand": product.get('brand', 'N/A'),
                    "reviews": len(self.result.get('reviews', [])),
                    "search_matches": self.result.get('matches_found', 'N/A')
                }
            }
            
            # Step 2: Sentiment Analysis
            yield {"type": "step", "step": 2, "status": "running", "message": "Analyzing customer reviews for sentiment..."}
            sentiment = self.analyze_sentiment()
            yield {
                "type": "step",
                "step": 2,
                "status": "completed",
                "message": f"Sentiment: {sentiment['overall_sentiment']} (score: {sentiment['avg_score']})",
                "data": {
                    "overall": sentiment['overall_sentiment'],
                    "score": sentiment['avg_score'],
                    "reviews_analyzed": sentiment['review_count_analyzed'],
                    "discrepancies": len(sentiment.get('discrepancies', []))
                }
            }
            
            # Step 3: Generate Prompt
            yield {"type": "step", "step": 3, "status": "running", "message": "Creating DALL-E prompt from complaints..."}
            dalle_prompt = self.generate_dalle_prompt()
            yield {
                "type": "step",
                "step": 3,
                "status": "completed",
                "message": f"Prompt created ({len(dalle_prompt)} chars)",
                "data": {"prompt_preview": dalle_prompt}
            }
            
            # Step 4: Generate Images
            yield {"type": "step", "step": 4, "status": "running", "message": "Generating images with DALL-E 3 & DALL-E 2..."}
            images = self.generate_images()
            successful_images = [i for i in images if 'filename' in i]
            yield {
                "type": "step",
                "step": 4,
                "status": "completed",
                "message": f"Generated {len(successful_images)} images",
                "data": {
                    "images": [{"model": i.get('model'), "filename": i.get('filename')} for i in successful_images]
                }
            }
            
            # Step 5: Vision Analysis
            yield {"type": "step", "step": 5, "status": "running", "message": "Analyzing images with GPT-4o Vision..."}
            vision = self.analyze_images()
            yield {
                "type": "step",
                "step": 5,
                "status": "completed",
                "message": f"Best image: #{vision.get('best_image', 'N/A')} (scores: {vision.get('image1_score', 'N/A')} vs {vision.get('image2_score', 'N/A')})",
                "data": vision
            }
            
            # Step 6: Save Results
            yield {"type": "step", "step": 6, "status": "running", "message": "Saving results to JSON and CSV..."}
            output_file = self.save_results()
            yield {
                "type": "step",
                "step": 6,
                "status": "completed",
                "message": "Results saved successfully",
                "data": {"output_file": output_file}
            }
            
            # Final result
            clean_result = self._clean_result_for_json(self.result)
            yield {
                "type": "complete",
                "success": True,
                "result": clean_result,
                "steps": self.steps
            }
            
        except Exception as e:
            self.log_step("Workflow Error", "failed", {"error": str(e)})
            yield {
                "type": "error",
                "success": False,
                "error": str(e),
                "steps": self.steps
            }

