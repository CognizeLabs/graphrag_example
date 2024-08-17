
from flask import Flask, request, jsonify, render_template
import networkx as nx
from transformers import RagTokenizer, RagTokenForGeneration
import torch

# Initialize the Flask app
app = Flask(__name__)

# Build the medical knowledge graph
def build_medical_knowledge_graph():
    G = nx.DiGraph()
    G.add_edge("Diabetes", "Insulin", relation="treated_by")
    G.add_edge("Heart Disease", "Exercise", relation="prevented_by")
    G.add_node("Aspirin", type="Medication")
    G.add_edge("Aspirin", "Blood Thinning", relation="used_for")
    G.add_edge("Aspirin", "Heart Attack", relation="prevents")
    G.add_edge("Cancer", "Chemotherapy", relation="treated_by")
    G.add_edge("Asthma", "Inhaler", relation="managed_by")
    return G

graph = build_medical_knowledge_graph()

# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

def get_answers(question, graph):
    inputs = tokenizer(question, return_tensors="pt")
    output = model.generate(**inputs)
    answer_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = get_answers(question, graph)
        return render_template('index.html', question=question, answer=answer)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    