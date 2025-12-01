"""
Seller Quality Optimizer - Flask Web Application
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, Response
from dotenv import load_dotenv
from workflow import WorkflowEngine

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

app = Flask(__name__, template_folder='templates', static_folder='output')

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "kaggle-dataset"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global workflow instance (reset per run)
current_workflow = None

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def run_workflow():
    """Execute the full workflow with streaming progress"""
    global current_workflow
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return jsonify({"success": False, "error": "OPENAI_API_KEY not set in .env file"})
    
    def generate():
        global current_workflow
        try:
            current_workflow = WorkflowEngine(
                api_key=api_key,
                data_dir=str(DATA_DIR),
                output_dir=str(OUTPUT_DIR)
            )
            
            # Run workflow with progress callbacks
            for progress in current_workflow.run_full_workflow_with_progress():
                yield f"data: {json.dumps(progress)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/run_simple', methods=['POST'])
def run_workflow_simple():
    """Execute the full workflow (non-streaming fallback)"""
    global current_workflow
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return jsonify({"success": False, "error": "OPENAI_API_KEY not set in .env file"})
    
    try:
        current_workflow = WorkflowEngine(
            api_key=api_key,
            data_dir=str(DATA_DIR),
            output_dir=str(OUTPUT_DIR)
        )
        result = current_workflow.run_full_workflow()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/status')
def get_status():
    """Get current workflow status"""
    global current_workflow
    if current_workflow:
        return jsonify({
            "steps": current_workflow.steps,
            "result": current_workflow.result
        })
    return jsonify({"steps": [], "result": {}})

@app.route('/api/outputs')
def list_outputs():
    """List all output files"""
    outputs = []
    for f in OUTPUT_DIR.glob("analysis_*.json"):
        with open(f) as file:
            data = json.load(file)
            outputs.append({
                "filename": f.name,
                "asin": data.get("asin"),
                "title": data.get("product_title", "")[:50],
                "analyzed_at": data.get("analyzed_at")
            })
    outputs.sort(key=lambda x: x.get("analyzed_at", ""), reverse=True)
    return jsonify(outputs)

@app.route('/api/csv')
def get_csv_data():
    """Get CSV data as JSON"""
    import pandas as pd
    csv_path = OUTPUT_DIR / "analysis_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict('records'))
    return jsonify([])

@app.route('/api/output/<filename>')
def get_output(filename):
    """Get a specific output file"""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "File not found"}), 404

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve generated images"""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    PORT = 5050
    print("\n" + "="*60)
    print("  SELLER QUALITY OPTIMIZER - Agentic Workflow")
    print("="*60)
    print(f"\n  Data Directory: {DATA_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"\n  Open http://localhost:{PORT} in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, port=PORT)

