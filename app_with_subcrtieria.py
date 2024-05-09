from flask import Flask, request, render_template, redirect, url_for, flash, session
import numpy as np
from fractions import Fraction
import json
import psycopg2
import numpy as np
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash
import re
from collections import OrderedDict
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

weight_dic = dict
# oussamaaaaa
# ededededed
app = Flask(__name__)
app.secret_key = 'oLjJcqqvOY'
num = 0
# Custom filter to mimic enumerate function in Jinja2 templates
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)


@app.route('/sub_cri_results')
def sub_cri_results():
    global nom_pro
    total_weights = []
    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()

    # Calculate total weight for each criteria
    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  # Sum of weights for the current criteria
        total_weights.append(total_weight)
    return render_template('sub_cri_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights,nom_pro=nom_pro)


@app.route('/cri_results')
def cri_results():

    total_weights = []
    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()

    # Calculate total weight for each criteria
    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  # Sum of weights for the current criteria
        total_weights.append(total_weight)
    return render_template('cri_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights)

# Add the zip function to Jinja's global namespace
app.jinja_env.globals['zip'] = zip


# Register the custom filter as a global function
app.jinja_env.globals['jinja2_enumerate'] = jinja2_enumerate
app.jinja_env.globals['enumerate'] = enumerate


@app.route('/topsys')
def topsys():
    # You can render a template or return a response
    if request.method == 'POST':
        # Assuming the form data containing alternatives is named 'alternatives'
        alternatives = request.form.getlist('alternatives')
        return redirect(url_for('show_topsis_form', alternatives=alternatives))
    return render_template('topsys.html')
'''
@app.route('/topsyss')
def topsyss():
    # You can render a template or return a response
    #return render_template('topsyss.html')
'''
@app.template_filter('parse_json')
def parse_json(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None



class Criteria:
    def __init__(self, name):
        self.name = name
        self.sub_criteria = []
        self.matrix = None
        self.weights = None
        self.consistent = False
        self.CR = None
        self.ranking = None
        

    def __str__(self):
        return self.name

    def add_criteria(self, sub_criteria):
        self.sub_criteria = sub_criteria


    def calculate_weights(self, comparisons):
        matrix = np.array(comparisons)
        num_subcriteria = matrix.shape[0]
        weights = np.zeros(num_subcriteria)

        # Calculate weights using the eigenvalue method
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_index = eigvals.argmax()
        max_eigvec = eigvecs[:, max_index].real
        weights = max_eigvec / max_eigvec.sum()
        self.weights = weights
        return weights

    def calculate_rankings(self, weights):
        # Calculate rankings based on the weights
        rankings = np.argsort(-np.array(weights)) + 1
        return rankings

    def check_consistency(self):
        n = self.matrix.shape[0]
        eigvals = np.linalg.eigvals(self.matrix)
        lambda_max = np.max(np.real(eigvals))
        CI = (lambda_max - n) / (n - 1)
        RI_values = [0.52, 1.12, 1.12, 1.24, 1.32, 1.41, 1.45,1.49,1.51]  # Include 0 for one criterion

        RI = RI_values[n-1]
        print("dddddddddddddddd",RI)
        CR = CI / RI
        if RI == 0:
            raise ValueError("Random Index (RI) cannot be zero")
        CR = CI / RI
        self.consistent = CR < 0.1
        self.CR = CR
    
    def calculate_normalized_subcriteria_weights(self, weights):
        # Normalize the weights
        normalized_weights = weights / np.sum(weights)
        return normalized_weights

    def normalize_matrix(self):
        self.matrix = self.matrix / np.sqrt(np.sum(self.matrix ** 2, axis=0))
    
    

global_num_criteria = 0
class Project :
    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

class Product:
    def __init__(self): 
        self.criteria = []
        self.compared_pairs = set()
        self.matrix = None
        self.weights = None
        self.consistent = False
        self.CR = None
        self.ranking = None
        self.topsis_ranking = None
        self.ideal_solution =None
        self.negative_ideal_solution = None
        self.relative_closeness = None
        self.alternatives_scores = {}
        self.project_id = None


    def add_alternative_scores(self, alternative_name, scores):
        self.alternatives_scores[alternative_name] = scores

    '''@app.route('/process_topsis', methods=['POST'])
    def process_topsis():
        # Assuming you've sent the data in the format alternative_criterion
        processed_data = {}
        for key in request.form:
            alternative, criterion = key.split('_')
            if alternative not in processed_data:
                processed_data[alternative] = {}
            processed_data[alternative][criterion] = request.form[key]'''

    def calculate_rankings(self):
        self.ranking = np.argsort(-self.weights) + 1

    def add_criteria(self, criteria):
        self.criteria = criteria

    
    def compare_criteria(self, form_data):
        print("Inside compare_criteria method") 
        num_criteria = len(self.criteria)
        num=num_criteria
        self.matrix = np.ones((num_criteria, num_criteria))
        print("num_criteria",num_criteria)

        for i in range(num_criteria):
            for j in range(i + 1, num_criteria):
                key = f'c{i+1}c{j+1}'
                comparison_value = form_data.get(key)
                print("key",key)
                print("comparison_value",comparison_value)
                print(f"Comparing criteria {i+1} and {j+1}. Key: {key}, Value: {comparison_value}")
                if comparison_value is None or comparison_value == '':
                    flash(f"Comparison value between criteria {i+1} and {j+1} is missing.")
                    return False
                try:
                    comparison_value = float(comparison_value)
                except ValueError:
                    flash(f"Comparison value between criteria {i+1} and {j+1} is not a valid number.")
                    return False
                self.matrix[i, j] = comparison_value
                self.matrix[j, i] = 1 / comparison_value
                self.compared_pairs.add((min(i, j), max(i, j)))
                if (j, i) not in self.compared_pairs:
                    inverse_value = 1 / comparison_value
                    form_data[f'c{j+1}c{i+1}'] = str(inverse_value)
        return True

    def number_criteria(self):
        num = len(self.criteria)
        global_num_criteria = num
        return num
        

    def calculate_weights(self):
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        max_index = eigvals.argmax()
        max_eigvec = eigvecs[:, max_index].real
        self.weights = max_eigvec / max_eigvec.sum()

    def check_consistency(self):
        n = self.matrix.shape[0]
        eigvals = np.linalg.eigvals(self.matrix)
        lambda_max = np.max(np.real(eigvals))
        CI = (lambda_max - n) / (n - 1)
        RI_values = [0.52, 0.89, 1.12, 1.24, 1.32, 1.41, 1.45]
        RI = RI_values[n-1]
        if RI == 0:
            raise ValueError("Random Index (RI) cannot be zero")
        CR = CI / RI
        self.consistent = CR < 0.1
        self.CR = CR
    
    def perform_topsis(self):
        # Step 1: Normalize the decision matrix
        normalized_matrix = self.normalize_matrix(self.matrix)

        # Step 2: Determine the weighted normalized decision matrix
        weighted_normalized_matrix = self.calculate_weighted_matrix(normalized_matrix)

        # Step 3: Identify the ideal and negative ideal solutions
        self.ideal_solution = self.calculate_ideal_solution(weighted_normalized_matrix)
        self.negative_ideal_solution = self.calculate_negative_ideal_solution(weighted_normalized_matrix)

        # Step 4: Calculate the distance of each alternative from the ideal and negative ideal solutions
        self.distances = self.calculate_distances(weighted_normalized_matrix, self.ideal_solution, self.negative_ideal_solution)

        # Step 5: Calculate the relative closeness to the ideal solution for each alternative
        self.relative_closeness = self.calculate_relative_closeness(self.distances)

        # Step 6: Rank the alternatives based on their relative closeness values
        self.topsis_ranking = np.argsort(self.relative_closeness) + 1


    def normalize_matrix(self, matrix):
        norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))
        return norm_matrix

    def calculate_weighted_matrix(self, normalized_matrix):
        weighted_matrix = normalized_matrix * self.weights
        return weighted_matrix

    def calculate_ideal_solution(self, weighted_normalized_matrix):
        ideal_solution = np.max(weighted_normalized_matrix, axis=0)
        self.ideal_solution = ideal_solution
        return ideal_solution

    def calculate_negative_ideal_solution(self, weighted_normalized_matrix):
        negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)
        self.negative_ideal_solution = negative_ideal_solution
        return negative_ideal_solution

    def calculate_distances(self, weighted_normalized_matrix, ideal_solution, negative_ideal_solution):
        dist_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution) ** 2, axis=1))
        dist_negative_ideal = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution) ** 2, axis=1))
        return dist_ideal, dist_negative_ideal

    def calculate_relative_closeness(self, distances):
        dist_ideal, dist_negative_ideal = distances
        relative_closeness = dist_negative_ideal / (dist_ideal + dist_negative_ideal)
        return relative_closeness
    
    def determine_normalized_subcriteria_weightings(self):
        # Step 1: Normalize the Sub-Criteria Comparison Matrix
        normalized_matrix = self.normalize_matrix(self.matrix)

        # Step 2: Calculate the Weighted Normalized Sub-Criteria Matrix
        weighted_normalized_matrix = self.calculate_weighted_matrix(normalized_matrix)

        # Step 3: Sum the Columns of the Weighted Normalized Sub-Criteria Matrix
        self.normalized_weights = np.sum(weighted_normalized_matrix, axis=0)

        

    # Previous code...

    def calculate_weighted_matrix(self, normalized_matrix):
        weighted_matrix = normalized_matrix * self.weights
        return weighted_matrix

    def normalize_matrix(self, matrix):
        norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))
        return norm_matrix

class DatabaseManager:
    def __init__(self, dbname, user, password = "1234", host='localhost', port='5432'):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def insert_criteria_data(self, criteria_data):
        conn = psycopg2.connect(database=self.dbname, user=self.user, password=self.password,
                                host=self.host, port=self.port)
        cur = conn.cursor()
        id_pro=get_id_project(nom_pro)
        print("id project in criteria table ",id_pro)
        for criterion_data in criteria_data:
            criterion_name, subcriteria_names, subcriteria_comparisons, weights, consistent, CR, rankings, normalized_subcriteria_weights = criterion_data

            # Convert weights, rankings, and normalized_subcriteria_weights to lists
            weights_list = weights.tolist()
            rankings_list = rankings.tolist()
            # normalized_weights_list = normalized_subcriteria_weights.tolist()

            # Prepare subcriteria data
            subcriteria_data = []
            for name, comparison_row in zip(subcriteria_names, subcriteria_comparisons):
                subcriteria_dict = {"name": name, "comparisons": {}}
                for i, value in enumerate(comparison_row):
                    subcriteria_dict["comparisons"][f"c{i+1}"] = value
                subcriteria_data.append(subcriteria_dict)

            # Execute the SQL query to insert data into the database
            cur.execute("""
                INSERT INTO criteria(criterion_name, subcriteria_data, weights, consistent, consistency_ratio, ranking, normalized_subcriteria_weights,project_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s , %s)
            """, (criterion_name, json.dumps(subcriteria_data), json.dumps(weights_list),
                bool(consistent), CR, json.dumps(rankings_list), json.dumps(normalized_subcriteria_weights),id_pro)) 

        # Commit the transaction and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()



    def insert_result(self, criteria, comparisons, weights, consistent, CR, ranking, topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness,project_name):
        conn = psycopg2.connect(database=self.dbname, user=self.user, password=self.password,
                                host=self.host, port=self.port)
        cur = conn.cursor()
        
        weights_list = weights.tolist()
        ranking_list = ranking.tolist()
        comparisons_jsonb = []
        for i, row in enumerate(comparisons):
            comparison_dict = {}
            for j, value in enumerate(row):
                if i != j:
                    comparison_dict[f"c{i+1}c{j+1}"] = value
            comparisons_jsonb.append(comparison_dict)
        id_pro=get_id_project(nom_pro)
        print("id project",id_pro)
        cur.execute("""
            INSERT INTO Product(criteria, comparisons, weights, consistent, consistency_ratio, ranking, topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness,project_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s)
        """, (json.dumps(criteria), json.dumps(comparisons_jsonb), json.dumps(weights_list),
            bool(consistent), CR, json.dumps(ranking_list), json.dumps(topsis_ranking.tolist()), json.dumps(ideal_solution.tolist()), json.dumps(negative_ideal_solution.tolist()), json.dumps(relative_closeness.tolist()),id_pro))
        
        conn.commit()
        cur.close()
        conn.close()

    def insert_project(self,project_name):
        conn = psycopg2.connect(database=self.dbname, user=self.user, password=self.password,
                                host=self.host, port=self.port)
        print("insert name of project")
        print(project_name)
        cur = conn.cursor()
        cur.execute("INSERT INTO Project(name) VALUES (%s)", (project_name,))

        
        conn.commit()
        cur.close()
        conn.close()


def get_id_project(name1):
    conn = psycopg2.connect(database="ahp_topsys_resultat", user="postgres",
                            password="1234", host="localhost", port="5432")
    cur = conn.cursor()
    
    cur.execute("SELECT id_proj FROM Project WHERE name = %s", (name1,))
    all_ids = cur.fetchall()
    
    if all_ids:
        last_id = all_ids[-1][0]  # Selecting the last ID from the list of IDs
        print("Last ID in get_id_project:", last_id)
    else:
        last_id = None
    
    conn.close()
    return last_id



@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
    
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
nom_pro=None
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global nom_pro
        print("indeeeeeeeeeex rouuuuuuute")
        num_criteria = int(request.form['num_criteria'])
        name_project = (request.form['nom_projet'])
        print(num_criteria)
        print("nom du project",name_project)
        session['name_project'] = name_project
        nom_pro=name_project
        return redirect(url_for('add_criteria', num_criteria=num_criteria))
    return render_template('index.html')
    

@app.route('/add_criteria/<int:num_criteria>', methods=['GET', 'POST'])
def add_criteria(num_criteria):
    if request.method == 'POST':
        criteria_names = [request.form[f'c{i+1}'] for i in range(num_criteria)]
        num_subcriteria = [int(request.form[f'num_subcriteria_{i+1}']) for i in range(num_criteria)]
        nom_du_project = request.form['project_name']
        print("add criteria route , project_name:",nom_du_project)
        session['criteria_names'] = criteria_names
        session['num_subcriteria'] = num_subcriteria
        # session['project_name'] = project_name
        
        # Redirect to compare_subcriteria route to compare sub-criteria
        return redirect(url_for('compare_criteria'))
    return render_template('add_criteria.html', num_criteria=num_criteria)




@app.route('/compare_subcriteria', methods=['GET', 'POST'])
def compare_subcriteria():
    criteria_names = session.get('criteria_names', [])
    num_subcriteria = session.get('num_subcriteria', [])
    weights_list = []
    rankings_list = []
    CR_list = []
    normalized_subcriteria_data = []  # List to hold normalized sub-criteria data

    if request.method == 'POST':
        form_data = dict(request.form)
        db_manager = DatabaseManager("ahp_topsys_resultat", "postgres", "1234")
        print("FORM dataaaaaaaaa",form_data)

        # Prepare criteria_data list to pass to insert_criteria_data function
        criteria_data = []
        for i, criterion_name in enumerate(criteria_names):
           
            matrix = np.ones((num_subcriteria[i], num_subcriteria[i]))
            subcriteria_names = form_data.get(f'subcriteria_names_{i}').split('-')  # Split the input string by '-'

            print("subcriteria_names",subcriteria_names)
            print("num_subcriteria_for_criterion",num_subcriteria[i])
            for j in range(num_subcriteria[i]):
                for k in range(j + 1, num_subcriteria[i]):
                    subcriteria_comparison_key = f'c{i}s{j+1}c{i}s{k+1}'
                    comparison_value = form_data.get(subcriteria_comparison_key)
                    print("key",subcriteria_comparison_key)
                    print("VALUE",comparison_value)
                    if comparison_value is None or comparison_value == '':
                        flash(f"Comparison value between sub-criteria {j+1} and {k+1} is missing.")
                        return False
                    try:
                        comparison_value = float(comparison_value)
                    except ValueError:
                        flash(f"Comparison value between sub-criteria {j+1} and {k+1} is not a valid number.")
                        return False
                    matrix[j, k] = comparison_value
                    matrix[k, j] = 1 / comparison_value

                    

            # Calculate weights, rankings, and consistency ratio
            criteria = Criteria(criterion_name)
            criteria.matrix = matrix  # Set the matrix attribute
            weights = criteria.calculate_weights(matrix)
            rankings = criteria.calculate_rankings(weights)
            criteria.check_consistency()
            CR = criteria.CR
            
            # Calculate normalized sub-criteria weights
            normalized_subcriteria_weights = criteria.calculate_normalized_subcriteria_weights(weights)
            print("hahaha",normalized_subcriteria_weights.tolist())
            # Append the results to the respective lists
            weights_list.append(weights.tolist())
            rankings_list.append(rankings.tolist())
            CR_list.append(CR)
            
            # Prepare data for insertion into database
            criterion_data = (
                criterion_name,  # Criterion name
                subcriteria_names,  # Sub-criteria names
                matrix.tolist(),  # Sub-criteria comparisons
                weights,  # Weights
                criteria.consistent,  # Consistency
                CR,  # Consistency ratio
                rankings,  # Rankings
                normalized_subcriteria_weights.tolist()  #Normalized sub-criteria weights
            )
            criteria_data.append(criterion_data)

        # Call the insert_criteria_data function
        db_manager.insert_criteria_data(criteria_data)

        return redirect(url_for('sub_cri_results'))
        # Redirect to calculate_results route after processing all criteria
        #return redirect(url_for('calculate_results'))

    return render_template('compare_subcriteria.html', criteria_names=criteria_names, num_subcriteria=num_subcriteria, normalized_subcriteria_data=normalized_subcriteria_data)

@app.route('/enter_comparison', methods=['GET', 'POST'])
def enter_comparison():
    if request.method == 'POST':
        # Handle form submission
        # For example, retrieve the submitted comparison values from request.form
        # Store the comparison values in your data structure
        
        # Redirect to compare_subcriteria route after form submission
        return redirect(url_for('compare_subcriteria'))
    else:
        # Render the template for entering comparison
        # Make sure to pass any necessary data to the template, like criteria_names
        criteria_names = [...]  # Populate with your criteria names
        return render_template('enter_comparison.html', criteria_names=criteria_names)

@app.route('/compare_criteria', methods=['GET','POST'])
def compare_criteria():
    global global_num_criteria
    global nom_pro
    print("compare_criteria route accessed") 
    criteria_names = session.get('criteria_names', [])
    num_subcriteria = session.get('num_subcriteria', [])
    project_name = session.get('project_name')

    if request.method == 'POST':
        print("I'm inside post request")
        # Handle comparisons for both criteria and sub-criteria
        form_data = dict(request.form)
        product = Product()
        

        print("Form data", form_data)
        product.add_criteria(criteria_names)

        total_weights = []
        results = get_last_inserted_result()
        result_criterias = get_last_inserted_result_criteria()

        # Calculate total weight for each criteria
        for criteria_tuple in result_criterias:
            total_weight = sum(criteria_tuple[3])  # Sum of weights for the current criteria
            total_weights.append(total_weight)

        

        print("Before calling compare_criteria method")
        print("project_name:",nom_pro)
        project=Project(nom_pro)
        

        if not product.compare_criteria(form_data):
            return redirect(url_for('compare_criteria'))

        print("After calling compare_criteria method")
        product.calculate_weights()
        print("CRIIIIII",product.CR)
        product.check_consistency()
        product.calculate_rankings()
        global_num_criteria =len(product.criteria)
        print("number criteria",num)
        print("product.matrix",product.matrix)
        product.perform_topsis()  # Perform TOPSIS analysis
        print("product.topsys",product.topsis_ranking)
        print("product.topsys.ideal",product.ideal_solution)
        print("product.topsys.negative",product.negative_ideal_solution)
        print("product.topsys.relative",product.relative_closeness)
        
        # Insert results into the database
        db_manager = DatabaseManager("ahp_topsys_resultat", "postgres", "1234")
        db_manager.insert_project(project.name)


        db_manager.insert_result(criteria_names, product.matrix, product.weights,
                                  product.consistent, product.CR, product.ranking,
                                  product.topsis_ranking, product.ideal_solution,
                                  product.negative_ideal_solution, product.relative_closeness,project.name)

        
        flash("Comparison values saved successfully.")
        if all(form_data.get(f'c{i+1}c{j+1}') for i in range(len(criteria_names)) for j in range(i+1, len(criteria_names))):
            return redirect(url_for('cri_results'))
            #return redirect(url_for('compare_subcriteria'))

    return render_template('compare_criteria.html', criteria=criteria_names, num_criteria=len(criteria_names), num_subcriteria=num_subcriteria)

@app.route('/calculate_results')
def calculate_results():
    total_weights = []
    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()
    
    # Calculate total weight for each criteria
    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  # Sum of weights for the current criteria
        total_weights.append(total_weight)
    
    print("result_criterias", result_criterias)
    print("Total Weights:", total_weights)

    return render_template('calculate_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights)   

def get_last_inserted_result():
    conn = psycopg2.connect(database="ahp_topsys_resultat", user="postgres",
                            password="1234", host="localhost", port="5432")
    cur = conn.cursor()
    cur.execute("""
        SELECT criteria, comparisons, weights, consistent, consistency_ratio, ranking,
               topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness
        FROM Product ORDER BY id DESC LIMIT 1
    """)
    result = cur.fetchone()
    conn.close()
    return result


def get_last_inserted_result_criteria():
    global global_num_criteria
    number_criteria = global_num_criteria
    conn = psycopg2.connect(database="ahp_topsys_resultat", user="postgres",
                            password="1234", host="localhost", port="5432")
    cur = conn.cursor()
    id_pro=get_id_project(nom_pro)
    print("id project in get last inserted result criteria",id_pro)
    cur.execute("SELECT criterion_name,subcriteria_data ,subcriteria_comparisons, weights, consistent, consistency_ratio, ranking,normalized_subcriteria_weights FROM criteria WHERE project_id = %s", (id_pro,))
    result = cur.fetchall()
    conn.close()
    return result

DB_HOST = "localhost"
DB_NAME = "ahp_topsys_resultat"
DB_USER = "postgres"
DB_PASS = "1234"
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
 

 
@app.route('/login/', methods=['GET', 'POST'])
def login():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
 
        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        # Fetch one record and return result
        account = cursor.fetchone()
 
        if account:
            password_rs = account['password']
            print(password_rs)
            # If account exists in users table in out database
            if check_password_hash(password_rs, password):
                # Create session data, we can access this data in other routes
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                # Redirect to home page
                return redirect(url_for('home'))
            else:
                # Account doesnt exist or username/password incorrect
                flash('Incorrect username/password')
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
 
    return render_template('login.html')
  
@app.route('/register', methods=['GET', 'POST'])
def register():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
    
        _hashed_password = generate_password_hash(password)
 
        #Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        print(account)
        # If account exists show error and validation checks
        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!')
        elif not username or not password or not email:
            flash('Please fill out the form!')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into users table
            cursor.execute("INSERT INTO users (fullname, username, password, email) VALUES (%s,%s,%s,%s)", (fullname, username, _hashed_password, email))
            conn.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('register.html')
   
   
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
  
@app.route('/profile')
def profile(): 
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM users WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


 
def decimal_to_fraction(value):
    return Fraction.from_float(value).limit_denominator()  

# Register the custom filter function with Jinja2 environment
app.jinja_env.filters['fraction'] = decimal_to_fraction


@app.route('/topsyss', methods=['GET', 'POST'])
def show_topsis_form():
    if request.method == 'POST':
        # Assuming the form data containing alternatives is named 'alternatives'
        #alternatives = request.form.getlist('alternatives')
    # Assuming 'Criteria' and 'Product' classes are already populated with data
        #criteria_names = [c.name for c in Criteria.query.all()]
        #alternatives = [a.name for a in Product.query.all()]
        #num_alternatives = request.args.get('numAlternatives', default=0, type=int)
        #alternatives = [request.args.get(f'alternative{i}') for i in range(num_alternatives)]


        # Fake data to simulate weights for each criterion
        #criteria_weights = {c: np.random.rand() for c in criteria_names}

        #return render_template('topsyss.html', alternatives=alternatives, num_alternatives=num_alternatives)
        total_weights = []

        results = get_last_inserted_result()
        result_criterias = get_last_inserted_result_criteria()

        print("*******route topsyss product",results)
        print("******route topsyss criteria",result_criterias)
        # Calculate total weight for each criteria
        for criteria_tuple in result_criterias:
            total_weight = sum(criteria_tuple[3])  # Sum of weights for the current criteria
            total_weights.append(total_weight)


        num_alternatives = int(request.form.get('numAlternatives', 0))
        alternatives = [request.form.get(f'alternative{i}') for i in range(num_alternatives)]

        # Generate fake data for criteria weights as an example
        criteria_weights = {f'Criterion {i + 1}': round(np.random.rand(), 2) for i in
                            range(3)}  # Example for 3 criteria

        return render_template('topsyss.html', alternatives=alternatives, criteria_weights=criteria_weights,
                               results=results, result_criterias=result_criterias, total_weights=total_weights
        )

        # If GET request or no data, redirect back to form or show an empty form
    return render_template('topsyss.html', alternatives=[], criteria_weights={})
alternative_noms = {}
@app.route('/process_topsis', methods=['POST'])
def process_topsis():
    global weight_dic 
    global alternative_noms
    form_data = dict(request.form)
    print("FROM", form_data)
    num_alternatives = int(request.form.get('numAlternatives', 0))
    alternatives = [request.form.get(f'alternative{i}') for i in range(num_alternatives)]
    # Initialize dictionaries to store criteria weights and alternative values
    criteria_weights = {}
    alternative_values = {}
    
    # Extract sub-criteria weights and alternative values from form data
    for key, value in form_data.items():
        # Check if the key contains subcriterion weight data
        if key.startswith('subcriterion_weight_'):
            subcriterion_name = key.split('_')[-1]
            weight = float(value)
            criteria_weights[subcriterion_name] = weight
        # Check if the key contains alternative values
        elif key.startswith('alternative_values_'):
            parts = key.split('_')
            subcriterion_name = '_'.join(parts[2:-1])  # Extract subcriterion name
            alternative_name = parts[-1]  # Extract alternative name
            value = float(value)
            if subcriterion_name not in alternative_values:
                alternative_values[subcriterion_name] = {}
            alternative_values[subcriterion_name][alternative_name] = value
    
    print("criteria_weights",criteria_weights)
    print("alternative_values",alternative_values)
    alternative_values_array = np.array([[alternative_values[criterion][alternative] for alternative in alternative_values[criterion]] for criterion in criteria_weights])

    alternative_values_array = np.array([[alternative_values[criterion][alternative] for alternative in alternative_values[criterion]] for criterion in criteria_weights])

    # Normalize the alternative values for each criterion
    normalized_alternatives = alternative_values_array / np.sqrt((alternative_values_array ** 2).sum(axis=1))[:, np.newaxis]

    # Convert criteria weights to a numpy array
    criteria_weights_array = np.array(list(criteria_weights.values()))

    # Multiply each normalized value by its respective criterion weight using broadcasting
    weighted_normalized_decision_matrix = normalized_alternatives * criteria_weights_array[:, np.newaxis]
    
    result_criterias = get_last_inserted_result_criteria()
    print("Weighted Normalized Decision Matrix:")
    print(weighted_normalized_decision_matrix)
    subcriteria_list = list(alternative_values.keys())
    weighted_normalized_decision_matrix_dict = OrderedDict()
    for i, subcriterion in enumerate(subcriteria_list):
        weighted_normalized_decision_matrix_dict[subcriterion] = weighted_normalized_decision_matrix[i]

    alternative_names = {}
    for subcriterion_name, values in alternative_values.items():
        for alternative_name in values:
            if alternative_name not in alternative_names:
                alternative_names[alternative_name] = []
            alternative_names[alternative_name].append(subcriterion_name)


    print("Weighted Normalized Decision Matrix diccccc:",weighted_normalized_decision_matrix_dict)
    print("alternative_names",alternative_names)
    weight_dic = weighted_normalized_decision_matrix_dict 
    alternative_noms = alternative_names
    return render_template('results.html', criteria_weights=criteria_weights, 
                           weighted_normalized_decision_matrix=weighted_normalized_decision_matrix_dict, 
                           alternative_names=alternative_names)

# @app.route('/negative_postive_alter', methods=['POST'])
# def negative_postive_alter():
#     # Get the form data
#     criteria_weights = {}
#     weighted_normalized_decision_matrix = {}
#     maximize_minimize = {}

#     # Extract weights for each sub-criterion
#     for key, value in request.form.items():
#         if key.endswith('_weight'):
#             subcriterion = key[:-7]  # Remove '_weight' from the key to get sub-criterion name
#             criteria_weights[subcriterion] = value
    
#     # Extract values for each sub-criterion
#     for key, value in request.form.items():
#         if key.endswith('_value[]'):
#             subcriterion = key[:-7]  # Remove '_value[]' from the key to get sub-criterion name
#             if subcriterion not in weighted_normalized_decision_matrix:
#                 weighted_normalized_decision_matrix[subcriterion] = []
#             weighted_normalized_decision_matrix[subcriterion].append(value)

#     # Extract whether to maximize or minimize for each sub-criterion
#     for key, value in request.form.items():
#         if value == 'maximize' or value == 'minimize':
#             subcriterion = key.split('_')[1]  # Extract sub-criterion name
#             operation = value
#             maximize_minimize[subcriterion] = operation

#     print("Form Data:", request.form)
#     print("criteria_weights:", criteria_weights)
#     print("weighted_normalized_decision_matrix:", weighted_normalized_decision_matrix)
#     print("maximize_minimize:", maximize_minimize)

#     return render_template('topsis_results.html', criteria_weights=criteria_weights, weighted_normalized_decision_matrix=weighted_normalized_decision_matrix)

criteria_weights_sen = {}

results_sensitivity_sen ={}

import math




@app.route('/negative_postive_alter', methods=['POST'])
def negative_postive_alter():
    # Get the form data
    global alternative_noms
    global weight_dic
    global criteria_weights_sen 
    global results_sensitivity_sen 
    global alternative_ranks2
    criteria_weights = {}
    weighted_normalized_decision_matrix = defaultdict(list)
    maximize_minimize2 = {}
    alternative_values = defaultdict(dict)

    for key, value in request.form.items():
        if key.endswith('_weight'):
            # Extract sub-criterion name
            subcriterion = key.split('_')[0]
            criteria_weights[subcriterion] = float(value)
        elif key.endswith('_value'):
            # Extract sub-criterion and alternative names
            parts = key.split('_')
            subcriterion = parts[0]
            alternative_name = parts[1]
            # Add value to alternative_values dictionary
            alternative_values[alternative_name][subcriterion] = float(value)
            weighted_normalized_decision_matrix[subcriterion].append(float(value))
        elif key.startswith('maximize_') or key.startswith('minimize_'):
            # Extract sub-criterion name
            subcriterion = key.split('_')[1]
            maximize_minimize2[subcriterion] = value
    
    # Initialize A* and A- dictionaries
    A_star = {}
    A_minus = {}

    # Calculate A* and A- for each sub-criterion
    for subcriterion, values in weighted_normalized_decision_matrix.items():
        # Check if values are empty
        if len(values) == 0:
            print("Values list is empty for sub-criterion:", subcriterion)
            continue

        # Calculate A* and A- for the current sub-criterion
        if maximize_minimize2[subcriterion] == 'maximize':
            A_star[subcriterion] = max(values)
            A_minus[subcriterion] = min(values)
            if A_minus[subcriterion] > A_star[subcriterion]:
                A_star[subcriterion], A_minus[subcriterion] = A_minus[subcriterion], A_star[subcriterion]
        else:
            A_star[subcriterion] = min(values)
            A_minus[subcriterion] = max(values)

    # Calculate Euclidean distances for each alternative from the positive and negative ideal solutions
    positive_distances = {}
    negative_distances = {}

    for alternative, values in alternative_values.items():
        # Calculate distance from the positive ideal solution
        positive_distance = math.sqrt(sum((values[criterion] - A_star[criterion])**2 for criterion in values))
        positive_distances[alternative] = positive_distance

        # Calculate distance from the negative ideal solution
        negative_distance = math.sqrt(sum((values[criterion] - A_minus[criterion])**2 for criterion in values))
        negative_distances[alternative] = negative_distance

    # Print the extracted data and distances (for debugging)
    print("Form Data:", request.form)
    print("criteria_weights:", criteria_weights)
    print("weighted_normalized_decision_matrix:", weighted_normalized_decision_matrix)
    print("maximize_minimize2:", maximize_minimize2)
    print("Alternative Values:", alternative_values)
    print("A_star:", A_star)
    print("A_minus", A_minus)
    print("Positive Distances:", positive_distances)
    print("Negative Distances:", negative_distances)
    print("alternative_noms:", alternative_noms)

    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()
    relative_closeness_coefficients = {}

    for alternative in positive_distances:
        positive_distance = positive_distances[alternative]
        negative_distance = negative_distances[alternative]

        # Calculate relative closeness coefficient
        relative_closeness_coefficient = negative_distance / (positive_distance + negative_distance)
        
        # Store the relative closeness coefficient for the alternative
        relative_closeness_coefficients[alternative] = relative_closeness_coefficient

    # Print or use the relative closeness coefficients as needed
    print("Relative Closeness Coefficients:", relative_closeness_coefficients)
    ranked_alternatives = sorted(relative_closeness_coefficients.items(), key=lambda x: x[1], reverse=True)
    print("Ranked Alternatives ",ranked_alternatives)
    # Get the optimal alternative (the one with the highest relative closeness coefficient)
    optimal_alternative, optimal_closeness_coefficient = ranked_alternatives[0]
    alternative_ranks = {}

    print("Ranked Alternatives (Descending Order):")
    for rank, (alternative, closeness_coefficient) in enumerate(ranked_alternatives, start=1):
        print("Rank", rank, "-", alternative, ": Closeness Coefficient =", closeness_coefficient)
        alternative_ranks[alternative] = rank


    # Now you have 15 cases of sensitivity analysis stored in the 'results' list
    # Get the optimal alternative (the one with the highest relative closeness coefficient)
    optimal_alternative, optimal_closeness_coefficient = ranked_alternatives[0]
    print("\nOptimal Alternative:", optimal_alternative)
    print("Alternative Ranks:", alternative_ranks)
    # Iterate over each subcriterion except the first one
    results_sensitivity = []
    subcriteria = list(criteria_weights.keys())
    print("length of subcriteria:", len(subcriteria))
        # Case 1: Constant weights
    results_sensitivity.append({
        "criteria_exchange": ("Constant", "No Change"),
        "criteria_weights": criteria_weights.copy()  # Copy original weights
    })

    # Iterate over each subcriterion except the first one
    for j in range(1, len(subcriteria)):
        # Create a copy of the criteria weights
        modified_weights = dict(criteria_weights)
        
        # Exchange the weight of the first subcriterion with the weight of the current subcriterion
        modified_weights[subcriteria[0]], modified_weights[subcriteria[j]] = modified_weights[subcriteria[j]], modified_weights[subcriteria[0]]
        print("{", modified_weights[subcriteria[0]], ",", modified_weights[subcriteria[j]], "}")
        # Store the results for this case
        case_results = {
            "criteria_exchange": (subcriteria[0], subcriteria[j]),
            "criteria_weights": modified_weights
        }
        

        results_sensitivity.append(case_results)

    

    # Row with equal weights
    equal_weight = {subcriterion: round(1 / len(subcriteria), 4) for subcriterion in subcriteria}
    results_sensitivity.append({
        "criteria_exchange": ("Equal", "Weights"),
        "criteria_weights": equal_weight
    })
    
    for idx, case_result in enumerate(results_sensitivity, 1):
        print(f"Case {idx}: {case_result['criteria_exchange']}")
        print("Criteria Weights:", case_result.get('criteria_weights'))
        if 'ranked_alternatives' in case_result:
            print("Ranked Alternatives:", case_result['ranked_alternatives'])


        
    # Print the results
    print("*****************************************************")
    print("results_sensitivity:", results_sensitivity)
    # Analyse des variations de poids des critères et calcul des classements des alternatives pour chaque cas de sensibilité
        # Créer une liste pour stocker les noms des alternatives
    # Créer une liste pour stocker les noms des cas de sensibilité et les classements correspondants



    criteria_weights_sen = criteria_weights 
    results_sensitivity_sen = results_sensitivity
    alternative_noms=alternative_values
    alternative_ranks2=alternative_ranks

 


    
    return render_template('topsis_results.html', 
                           criteria_weights=criteria_weights, 
                           weighted_normalized_decision_matrix=weighted_normalized_decision_matrix,
                           results=results, result_criterias=result_criterias, maximize_minimize2=maximize_minimize2,
                           A_star=A_star, A_minus=A_minus,
                           alternative_values=alternative_values,
                           positive_distances=positive_distances,
                           negative_distances=negative_distances,
                           alternative_noms= alternative_noms,
                           weight_dic=weight_dic,
                           positive_distance=positive_distances,
                           negative_distance=negative_distances,
                           relative_closeness_coefficient=relative_closeness_coefficients,
                           alternative_ranks=alternative_ranks,
                           results_sensitivity=results_sensitivity)


@app.route('/graph_draw')
def graph_draw():
    sensitivity_cases = []
    rankings_per_case = []

    # Parcourir chaque cas de sensibilité
    for case in results_sensitivity_sen:
        # Obtenir les noms des cas de sensibilité
        sensitivity_cases.append(f"{case['criteria_exchange'][0]} → {case['criteria_exchange'][1]}")

        # Obtenir les poids des critères pour ce cas
        case_weights = case['criteria_weights']

        # Initialiser les dictionnaires pour les distances positives et négatives
        positive_distances = {}
        negative_distances = {}

        # Calculer A_star et A_minus pour ce cas de sensibilité
        A_star = {criterion: max(alternative_noms[alternative][criterion] for alternative in alternative_noms) for criterion in case_weights}
        A_minus = {criterion: min(alternative_noms[alternative][criterion] for alternative in alternative_noms) for criterion in case_weights}

        # Calculer les distances positives et négatives pour chaque alternative avec les nouveaux poids des critères
        for alternative, values in alternative_noms.items():
            # Initialiser les variables pour stocker les distances positives et négatives
            positive_distance = 0
            negative_distance = 0

            # Calculer la distance positive
            for criterion, weight in case_weights.items():
                positive_distance += weight * (values[criterion] - A_star[criterion]) ** 2

            # Calculer la distance négative
            for criterion, weight in case_weights.items():
                negative_distance += weight * (values[criterion] - A_minus[criterion]) ** 2

            # Prendre la racine carrée des sommes des carrés pour obtenir les distances
            positive_distance = math.sqrt(positive_distance)
            negative_distance = math.sqrt(negative_distance)

            # Stocker les distances positives et négatives pour l'alternative
            positive_distances[alternative] = positive_distance
            negative_distances[alternative] = negative_distance

        # Recalculer les classements des alternatives avec les nouvelles distances positives et négatives
        relative_closeness_coefficients = {}
        for alternative in positive_distances:
            positive_distance = positive_distances[alternative]
            negative_distance = negative_distances[alternative]

            # Calculer le coefficient de proximité relative
            relative_closeness_coefficient = negative_distance / (positive_distance + negative_distance)
            relative_closeness_coefficients[alternative] = relative_closeness_coefficient

        # Classer les alternatives en fonction du coefficient de proximité relative
        ranked_alternatives = sorted(relative_closeness_coefficients.items(), key=lambda x: x[1], reverse=True)
        rankings = [rank for rank, (alternative, closeness_coefficient) in enumerate(ranked_alternatives, start=1)]
        rankings_per_case.append(rankings)

        # Stocker les noms des alternatives
        alternatives = [alternative for alternative, _ in ranked_alternatives]

        # Afficher les classements des alternatives pour ce cas de sensibilité
        print("Classements des alternatives pour le cas de sensibilité:", case['criteria_exchange'])
        for rank, (alternative, closeness_coefficient) in enumerate(ranked_alternatives, start=1):
            print("Rang", rank, "-", alternative, ": Coefficient de proximité relative =", closeness_coefficient)

    # Créer un graphique en lignes pour visualiser les changements de classement pour chaque alternative
    plt.figure(figsize=(12, 8))
    for alternative in alternative_noms:
        alternative_rankings = []
        for i, rankings in enumerate(rankings_per_case):
            alternative_rankings.append(rankings[alternative_ranks2[alternative] - 1])
        plt.plot(range(len(rankings_per_case)), alternative_rankings, marker='o', label=alternative)

    # Définir les étiquettes de l'axe des abscisses comme les noms des cas de sensibilité
    plt.xticks(range(len(rankings_per_case)), sensitivity_cases, rotation=45, ha='right')
    plt.xlabel('Cas de sensibilité')
    plt.ylabel('Classement')
    plt.title('Changements des classements des alternatives pour chaque cas de sensibilité')
    plt.legend(loc='upper right')
   # Inverser l'axe y pour afficher les barres de haut en bas
    plt.tight_layout()
    plt.show()
    return "Graph"

@app.route('/sensitivity')
def sensitivity():
    print("sensitivity roooooooooooute ***********")
    print("criteria_weights_sen",criteria_weights_sen)
    print("results_sensitivity_sen",results_sensitivity_sen)


    return render_template('sensitivity.html',criteria_weights_sen=criteria_weights_sen,
    results_sensitivity_sen=results_sensitivity_sen)

@app.route('/get_subcriteria_names')
def get_subcriteria_names():
    # Assuming you have a function to retrieve subcriteria names
    subcriteria_names = get_subcriteria_names_from_database()  # Replace this with your own function

    # Returning the subcriteria names as JSON
    return jsonify({'subcriteria_names': subcriteria_names})
if __name__ == '__main__':
    app.run(debug=True)