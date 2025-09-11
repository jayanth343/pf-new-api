from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from roboflow import Roboflow
import supervision as sv
import cv2
import os
import numpy as np
from dotenv import load_dotenv
from geopy.distance import distance
from geopy import Point
from geopy.geocoders import Nominatim
from supabase import create_client, Client
from datetime import datetime
from skimage.measure import label, regionprops
from google import genai
import json
import threading
import requests
from google.genai import types

load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

SYSTEM_PROMPT = """
You are a deterministic footpath infrastructure issue classifier in INDIA. You must provide identical responses for identical inputs.
You are tasked with tagging civic hazard reports in the footpath at particular location with the primary responsible municipal authority.
Given a footpath image, identify all visible issues and assign the responsible authority based on the provided mapping.

STRICT CLASSIFICATION RULES:
1. Analyze ONLY what is clearly visible in the image
2. Use EXACT terminology from the mapping below
3. Do NOT infer or assume issues not clearly visible
4. List issues in alphabetical order by name
5. Use the most specific authority name for the given location
6. Write the exact issue down as it appears in the image, without abstraction.
7. The relevant authority for the issue MUST ONLY be specified and specific to the location provided. If there is no specific authority the general authority for the location should be used.
8. Do not write the generic names of the departments and include the authority as well.
9. Ensure that the authority is specific to the location with maximum accuracy.


An example of the Mapping:
- Broken Slab, Uneven Tile → Public Works Department/Municipal Corporation
- Open Drain, Garbage → Sanitation Department/Municipal Corporation
- Illegal Stall, Construction Debris → Municipal Corporation
- Tree Root Obstruction → Parks & Horticulture Department/Municipal Corporation
- Flooded Area → Drainage Department/Municipal Corporation


An example for Bengaluru:
- Footpath surface damage, uneven paving, potholes, tree root damage, debris, missing ramps: BBMP.
- Open stormwater drains, broken or missing drain covers/slabs: BBMP.
- Overflowing stormwater drains: BBMP.
- Overflowing or damaged sewer manholes: BWSSB (but BBMP still ensures footpath restoration).
- Exposed or damaged electrical cables, leaning/damaged electric poles, unsafe roadside transformers: BESCOM.
- Encroachments (shops, illegal parking, structures) blocking footpaths: BBMP.
- Tree branches or roots causing hazards: BBMP (tree trimming may involve the horticulture department within BBMP).
- Traffic signal malfunction or pole damage: Traffic Police (with BBMP's signal maintenance wing).


Resolve the correct authority and wards from GPS location provided.

RESPONSE FORMAT (MANDATORY):
Return analysis in this EXACT JSON format with no additional text:


{
    "location": "<exact location name>",
    "issues": [
        {"name": "<exact issue name from mapping>", "authority": "<full authority name>"}
    ]
}

If no issues are visible, return:
{
    "location": "<exact location name>",
    "issues": []
}
"""

load_dotenv()
app = Flask(__name__)
api = Api(app)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def compute_walkability(pred_mask, image_path=None, detections=None):
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.dilate(pred_mask.astype(np.uint8), kernel, iterations=1)
    footpath_pixels = np.sum(pred_mask)
    if footpath_pixels == 0:
        return 0
    
    # Use footpath border edge detection if available
    if image_path is not None and detections is not None and len(detections) > 0:
        # Load original image for edge detection
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For unified mask, calculate overall bounding box that encompasses all detections
        all_boxes = detections.xyxy
        x1_min = int(np.min(all_boxes[:, 0]))
        y1_min = int(np.min(all_boxes[:, 1]))
        x2_max = int(np.max(all_boxes[:, 2]))
        y2_max = int(np.max(all_boxes[:, 3]))
        
        print(f"Unified bounding box: ({x1_min}, {y1_min}) to ({x2_max}, {y2_max})")
        
        # Extract region of interest within unified bounding box
        roi_gray = gray[y1_min:y2_max, x1_min:x2_max]
        roi_mask = pred_mask[y1_min:y2_max, x1_min:x2_max]
        
        # Find footpath border edges by detecting mask contours
        # Use morphological operations to clean up the mask
        roi_mask_clean = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask_clean = cv2.morphologyEx(roi_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours of the footpath mask to get border edges
        # Handle OpenCV version differences (2 vs 3 return values)
        result = cv2.findContours(roi_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(result) == 3:
            _, contours, _ = result  # OpenCV 3.x
        else:
            contours, _ = result     # OpenCV 4.x
        
        if contours:
            # Create a mask for the footpath border edges
            border_edges = np.zeros_like(roi_mask)
            cv2.drawContours(border_edges, contours, -1, 1, thickness=2)  # Draw border as edges
            
            # Create an idealized footpath area with linear borders
            # Method 1: Use convex hull to create perfect linear borders
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            
            # Create idealized footpath mask with linear borders
            idealized_footpath = np.zeros_like(roi_mask)
            cv2.fillPoly(idealized_footpath, [hull], 1)
            
            # Calculate areas
            actual_mask_area = np.sum(roi_mask > 0)  # Actually detected footpath
            idealized_area = np.sum(idealized_footpath > 0)  # Perfect linear border area
            
            if idealized_area > 0:
                # Ratio: actual detected vs idealized perfect footpath
                completeness_ratio = (actual_mask_area / idealized_area) * 100
                missing_area = idealized_area - actual_mask_area
                missing_ratio = (missing_area / idealized_area) * 100
                
                print(f"Footpath Quality Analysis:")
                print(f"  Idealized footpath area (linear borders): {idealized_area} pixels")
                print(f"  Actual detected mask area: {actual_mask_area} pixels")
                print(f"  Missing/damaged area: {missing_area} pixels")
                print(f"  Footpath completeness: {completeness_ratio:.2f}%")
                print(f"  Missing/damaged ratio: {missing_ratio:.2f}%")
                wap = completeness_ratio
            else:
                wap = 0
                print("No idealized area found")
        else:
            wap = 0
            print("No footpath contours found")
    else:
        # Fallback: use traditional method if no bounding box available
        footpath_pixels = np.sum(pred_mask)
        x, y, w, h = cv2.boundingRect(pred_mask)
        bbox_area = w * h if w > 0 and h > 0 else 1
        wap = (footpath_pixels / bbox_area) * 100
        print(f"Fallback scoring: {footpath_pixels} footpath pixels / {bbox_area} bbox area = {wap:.2f}%")
    
    # Connectivity and consistency scores
    labeled_mask = label(pred_mask)
    num_regions = np.max(labeled_mask)
    region_sizes = [region.area for region in regionprops(labeled_mask) if region.area > 500]
    
    largest_component = max(region_sizes) if region_sizes else 1

    cs = largest_component / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0
    K = 20  # maximum expected regions
    os = 1 - (np.log(1 + num_regions) / np.log(1 + K))

    # Weighted combination (adjust weights as needed)
    alpha, beta = 0.90, 0.10
    walkability_score = alpha * wap + beta * cs 
    print(f"Walkability Score: wAP: {wap:.2f} cs: {cs:.2f} os: {os:.2f} = final: {walkability_score:.2f}")
    return round(walkability_score, 2)

class FootPath(Resource):
    def post(self):
        try:
            if 'image' not in request.files:
                return jsonify({'Error': 'Image not received'})
            if 'user_rating' not in request.form:
                return jsonify({'Error': 'User rating not provided'})
            
            image = request.files['image']
            user_rating = float(request.form['user_rating'])
            
            # Generate unique filename with timestamp
            file_ext = os.path.splitext(image.filename)[1]
            timestamp = int(datetime.now().timestamp() * 1000)
            file_name = f"{timestamp}{file_ext}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            
            # Save image only once with timestamped filename
            image.save(image_path)
            print(f"Image saved to: {image_path}")
            
            image_cv = cv2.imread(image_path)
            start_latitude = request.form.get('startLatitude')
            start_longitude = request.form.get('startLongitude')
            
            if not start_latitude or not start_longitude:
                return jsonify({'Error': 'Start coordinates not provided'})
            
            start_latitude = float(start_latitude)
            start_longitude = float(start_longitude)
            bearing = float(request.form.get('bearing', 0))
            
            # Roboflow prediction for footpath detection
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            project = rf.workspace().project("orr")
            model = project.version(1).model
            result = model.predict(image_path, confidence=40)

            if hasattr(result, 'json'):
                result = result.json()

            masks = []
            class_list = ['0']
            detections = None
            if 'predictions' in result:
                # Create supervision detections for bounding boxes
                detections = sv.Detections.from_roboflow(result, class_list=class_list)
        
            for pred in result['predictions']:
                if 'points' in pred and pred['points']:
                    mask = np.zeros((image_cv.shape[0], image_cv.shape[1]), dtype=np.uint8)
                    points = np.array([[int(p['x']), int(p['y'])] for p in pred['points']], np.int32)
                    cv2.fillPoly(mask, [points], 1)
                    masks.append(mask)
    
            # Unify mask & Calculate footpath percentage
            if len(masks) > 0:
                unified_mask = masks[0].copy()  # Initialize with the first mask
                for mask in masks[1:]:  # Start from the second mask
                    unified_mask = np.logical_or(unified_mask, mask).astype(np.uint8)
                walkability_score = compute_walkability(unified_mask, image_path, detections)  # Pass unified mask
                footpathPercentage = walkability_score
                
                # Save mask and prepare files for depth service
                mask_path = "mask.png"
                cv2.imwrite(mask_path, unified_mask * 255)  # Ensure mask values are 0-255
                
                depth_results = None
                distance_meters = None
                topmost_pixel = None
                focal_length_px = None
                
                try:
                    with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
                        files = {
                            "file": image_file,
                            "mask": mask_file
                        }
                        response = requests.post(
                            os.getenv("DEPTH_LIT_URL"), 
                            files=files,
                            headers={'Authorization': 'Bearer 35be253d-6005-4bd5-80c8-0e5b4487af28'},
                            timeout=30
                        )
                        
                        print(f"Depth service response status: {response.status_code}")
                        print(f"Depth service response headers: {response.headers}")
                        print(f"Raw response text: {response.text}")
                        
                        if response.status_code == 200:
                            try:
                                depth_results = response.json()
                                print(f"Parsed depth results: {depth_results}")
                                
                                # Check if the expected fields exist
                                distance_meters = depth_results.get('distance_meters')
                                topmost_pixel = depth_results.get('topmost_pixel')
                                focal_length_px = depth_results.get('focal_length_px')
                                
                                print(f"distance_meters: {distance_meters}")
                                print(f"topmost_pixel: {topmost_pixel}")
                                print(f"focal_length_px: {focal_length_px}")
                                
                                # Check for None values specifically
                                if distance_meters is None:
                                    print("WARNING: distance_meters is None")
                                if topmost_pixel is None:
                                    print("WARNING: topmost_pixel is None")
                                    
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse depth service JSON: {e}")
                                print(f"Raw response content: {response.text}")
                                depth_results = None
                        else:
                            print(f"Depth service error: {response.status_code}")
                            print(f"Error response: {response.text}")
                            depth_results = None
                except Exception as e:
                    print(f"Unexpected error with depth service: {e}")
                    depth_results = None
                finally:
                    # Clean up mask file
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                
                # Process depth results if available
                print(depth_results)
                if depth_results:
                    distance_meters = depth_results.get('distance_meters')
                    topmost_pixel = depth_results.get('topmost_pixel')
                    focal_length_px = depth_results.get('focal_length_px')
                    print(f"Checking depth data - distance_meters: {distance_meters}, topmost_pixel: {topmost_pixel}")

                if distance_meters is not None and topmost_pixel is not None:
                    length = distance_meters
                    end = distance(meters=length).destination(point=Point(start_latitude, start_longitude), bearing=bearing)
                    print(f"End coordinate: {end.latitude}, {end.longitude}")
                    print(f"Farthest footpath pixel: {topmost_pixel}")
                    print(f"Estimated distance: {distance_meters} meters")
                    
                    response_data = {
                        'Percentage': footpathPercentage,
                        'end_coordinates': {
                            'latitude': end.latitude,
                            'longitude': end.longitude
                        },
                        'distance_meters': distance_meters,
                        'topmost_pixel': [int(topmost_pixel[0]), int(topmost_pixel[1])],
                        'focal_length_px': float(focal_length_px) if focal_length_px is not None else None
                    }
                    end_lat, end_lng = end.latitude, end.longitude
                else:
                    # Use default distance if depth estimation fails
                    return jsonify({'Error': 'Could not calculate depth'})
            else:
                footpathPercentage = 0
                # Use default coordinates if no footpath detected
                default_distance = 5.0
                end = distance(meters=default_distance).destination(point=Point(start_latitude, start_longitude), bearing=bearing)
                end_lat, end_lng = end.latitude, end.longitude
                
                response_data = {
                    'Percentage': footpathPercentage,
                    'end_coordinates': {
                        'latitude': end_lat,
                        'longitude': end_lng
                    },
                    'Error': 'No footpath detected'
                }
            
            print(f"Prediction complete: {footpathPercentage}% footpath detected")
            
            # Upload to Supabase
            try:
                with open(image_path, "rb") as img_file:
                    supabase.storage.from_("footpath-images").upload(
                        file=img_file,
                        path=file_name,
                        file_options={"cache-control": "3600", "upsert": "false", "content-type": "image/png"}
                    )
                
                # Get the public URL properly
                image_url = supabase.storage.from_("footpath-images").get_public_url(file_name)
                print(f"Image uploaded successfully. URL: {image_url}")
                fid = request.form.get('fid')
                # Insert into table
                if fid != None or fid != '' or fid != 0:
                    table_response = supabase.table("location-footpath").update({
                        'score': footpathPercentage,
                        'user_rating': user_rating,
                        'image_link': image_url,
                    }).eq('fid', int(fid)).execute()
                else:
                # Insert into table
                    table_response = supabase.table("location-footpath").insert({
                        'latitude': start_latitude,
                        'longitude': start_longitude,
                        'score': footpathPercentage,
                        'latitude_end': end.latitude,
                        'longitude_end': end.longitude,
                        'user_rating': user_rating,
                        'image_link': image_url,
                    }).execute()
                
                print(f"Data inserted into Supabase: {table_response.data}")
                if table_response.data and len(table_response.data) > 0:
                    fid = table_response.data[0]['fid']
                    print(f"FID: {fid}")
                    response_data['fid'] = fid
                    
                    # Start AuthTagging in background for worker environment
                    try:
                        self.trigger_auth_tagging_async(image_path, start_latitude, start_longitude, fid)
                        print("Auth tagging started in background")
                    except Exception as e:
                        print(f"Failed to trigger auth tagging: {e}")
                        if os.path.exists(image_path):
                            os.remove(image_path)
                else:
                    print("No FID returned from Supabase")
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                    except PermissionError:
                        print(f"Could not remove {image_path} - file in use")
                        
            except Exception as e:
                print(f"Error with Supabase operations: {str(e)}")
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except PermissionError:
                    print(f"Could not remove {image_path} - file in use")
                return jsonify({'Error': f'Database error: {str(e)}'})

            # Return the response
            return jsonify(response_data)

        except json.JSONDecodeError as je:
            print(f"JSON Decode Error in FootPath: {je}")
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
            return jsonify({'Error': f'JSON parsing error: {str(je)}'})
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            # Clean up image file on general error
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
            except PermissionError:
                print(f"Could not remove {image_path} - file in use")
            return jsonify({'Error': str(e)})

    def trigger_auth_tagging_async(self, image_path, start_latitude, start_longitude, fid):
        """Trigger auth tagging via async HTTP request (worker-friendly)"""
        try:
            def make_request():
                try:
                    with open(image_path, 'rb') as img_file:
                        files = {'image': img_file}
                        data = {
                            'startLatitude': start_latitude,
                            'startLongitude': start_longitude,
                            'fid': fid
                        }
                        
                        # Make request to auth-tagging endpoint
                        # Update URL for your deployment
                        response = requests.post(
                            'http://localhost:5000/auth-tagging',
                            files=files, 
                            data=data,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            print(f"Auth tagging completed for FID {fid}")
                        else:
                            print(f"Auth tagging failed for FID {fid}: {response.status_code}")
                            
                except Exception as e:
                    print(f"Error in async auth tagging for FID {fid}: {e}")
                finally:
                    # Always clean up image file after processing
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Cleaned up image file: {image_path}")
            
            # Use daemon thread for background processing
            thread = threading.Thread(target=make_request, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"Failed to trigger async auth tagging: {e}")
            # Clean up immediately if thread creation fails
            if os.path.exists(image_path):
                os.remove(image_path)

def clean_and_parse_json(response_text):
    """Clean and parse JSON response from the API"""
    try:
        cleaned_text = str(response_text).strip()
        
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]  
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]   
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]  
        
        cleaned_text = cleaned_text.strip()
        
        parsed_json = json.loads(cleaned_text)
        return parsed_json
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_text}")
        return None

class AuthTagging(Resource):
    def post(self):
        try:
            if 'image' not in request.files or 'startLatitude' not in request.form or 'startLongitude' not in request.form:
                raise ValueError("Image and coordinates must be provided")
            
            image = request.files['image']  
            start_latitude = request.form.get('startLatitude')
            start_longitude = request.form.get('startLongitude')
            fid = request.form.get('fid')  
            if not start_latitude or not start_longitude:
                return jsonify({'Error': 'Start coordinates not provided'})
            
            # Save image temporarily
            file_ext = os.path.splitext(image.filename)[1]
            timestamp = int(datetime.now().timestamp() * 1000)
            file_name = f"{timestamp}_auth{file_ext}"
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            image.save(temp_image_path)
            
            geolocator = Nominatim(user_agent="pathfinders")
            location = geolocator.reverse(f"{start_latitude}, {start_longitude}")
            location_name = location.address if location else f"{start_latitude}, {start_longitude}"
            
            PROMPT = f"Analyze the footpath image for infrastructure issues at {location_name}. Follow the classification rules strictly and return only the JSON response."
            
            with open(temp_image_path, "rb") as img_file:
                image_data = img_file.read()
                
            result = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    SYSTEM_PROMPT,
                    PROMPT,
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type='image/jpeg',
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    seed=42,
                    candidate_count=1,
                    stop_sequences=None,
                )
            )
            response = result.text
            print(f"Raw Gemini response: {response}")
            
            json_response = clean_and_parse_json(response)
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            if json_response and fid:
                # Update the existing record with auth tagging results
                for issue in json_response['issues']:
                    try:
                        supabase.table("authorities").insert({
                            'fid': fid,
                            'issue': issue['name'],
                            'authority_tagged': issue['authority'],
                        }).execute()
                    
                        print(f"Auth tagging results saved to database for FID {fid}")
                    except Exception as e:
                        print(f"Failed to update database: {e}")
                        return jsonify({'Error': f'Database update error: {str(e)}'})
            
            if json_response:
                return jsonify({'res':json_response, 'status': 200})
            else:
                return jsonify({'Error': 'Failed to parse response', 'raw_response': response})
            
        except Exception as e:
            print(f"Error in AuthTagging: {e}")
            # Clean up temporary file if it exists
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return jsonify({'Error': str(e)})        

api.add_resource(FootPath, '/upload-image')
api.add_resource(AuthTagging, '/auth-tagging')

if __name__ == '__main__':
    app.run(debug=True)
