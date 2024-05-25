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
from flask import send_file



app = Flask(__name__)
app.secret_key = 'oLjJcqqvOY'
num = 0
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)
def decimal_to_fraction(value):
    return Fraction.from_float(value).limit_denominator()  

app.jinja_env.filters['fraction'] = decimal_to_fraction


@app.route('/sub_cri_results')
def sub_cri_results():
    global nom_pro
    total_weights = []
    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()

    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  
        total_weights.append(total_weight)
    return render_template('sub_cri_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights,nom_pro=nom_pro)


@app.route('/cri_results')
def cri_results():

    total_weights = []
    results = get_last_inserted_result()
    result_criterias = get_last_inserted_result_criteria()

    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  
        total_weights.append(total_weight)
    return render_template('cri_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights)

app.jinja_env.globals['zip'] = zip


app.jinja_env.globals['jinja2_enumerate'] = jinja2_enumerate
app.jinja_env.globals['enumerate'] = enumerate


@app.route('/topsys')
def topsys():
    if request.method == 'POST':
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

class Result:
    def __init__(self, name):
        self.sensitivity = None


class Topsys:
    def __init__(self, name):
        self.alternative = None
        self.sensitivity = None


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

    
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_index = eigvals.argmax()
        max_eigvec = eigvecs[:, max_index].real
        weights = max_eigvec / max_eigvec.sum()
        self.weights = weights
        return weights

    def calculate_rankings(self, weights):
      
        rankings = np.argsort(-np.array(weights)) + 1
        return rankings

    def check_consistency(self):
        n = self.matrix.shape[0]
        eigvals = np.linalg.eigvals(self.matrix)
        lambda_max = np.max(np.real(eigvals))
        CI = (lambda_max - n) / (n - 1)
        RI_values = [0.52, 1.12, 1.12, 1.24, 1.32, 1.41, 1.45,1.49,1.51]  

        RI = RI_values[n-1]
        print("dddddddddddddddd",RI)
        CR = CI / RI
        if RI == 0:
            raise ValueError("Random Index (RI) cannot be zero")
        CR = CI / RI
        self.consistent = CR < 0.1
        self.CR = CR
    
    def calculate_normalized_subcriteria_weights(self, weights):
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
        normalized_matrix = self.normalize_matrix(self.matrix)

        weighted_normalized_matrix = self.calculate_weighted_matrix(normalized_matrix)

        self.ideal_solution = self.calculate_ideal_solution(weighted_normalized_matrix)
        self.negative_ideal_solution = self.calculate_negative_ideal_solution(weighted_normalized_matrix)

        self.distances = self.calculate_distances(weighted_normalized_matrix, self.ideal_solution, self.negative_ideal_solution)

        self.relative_closeness = self.calculate_relative_closeness(self.distances)

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
        normalized_matrix = self.normalize_matrix(self.matrix)

        weighted_normalized_matrix = self.calculate_weighted_matrix(normalized_matrix)

        self.normalized_weights = np.sum(weighted_normalized_matrix, axis=0)

    
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

            weights_list = weights.tolist()
            rankings_list = rankings.tolist()

            subcriteria_data = []
            for name, comparison_row in zip(subcriteria_names, subcriteria_comparisons):
                subcriteria_dict = {"name": name, "comparisons": {}}
                for i, value in enumerate(comparison_row):
                    subcriteria_dict["comparisons"][f"c{i+1}"] = value
                subcriteria_data.append(subcriteria_dict)

            cur.execute("""
                INSERT INTO criteria(criterion_name, subcriteria_data, weights, consistent, consistency_ratio, ranking, normalized_subcriteria_weights,project_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s , %s)
            """, (criterion_name, json.dumps(subcriteria_data), json.dumps(weights_list),
                bool(consistent), CR, json.dumps(rankings_list), json.dumps(normalized_subcriteria_weights),id_pro)) 

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
        print("user id",session['id'])
        id_user =session['id']
        print(project_name)
        cur = conn.cursor()
        cur.execute("INSERT INTO Project(name,user_id) VALUES (%s,%s)", (project_name,id_user))
        

        conn.commit()
        cur.close()
        conn.close()
    
    def insert_alternatives_and_sensitivity(self, alternatives, sensitivity_results):
        conn = psycopg2.connect(database=self.dbname, user=self.user, password=self.password,
                                host=self.host, port=self.port)
        cur = conn.cursor()
        
        
        cur.execute("""
                INSERT INTO Topsys(alternative,sensitivity)
                VALUES (%s, %s)
            """, (json.dumps(alternatives),json.dumps(sensitivity_results)))
        
        conn.commit()
        cur.close()
        conn.close()

    def insert_results(self,results_sensitivity,weighted_normalized_decision_matrix,A_star,A_minus,positive_distance,negative_distance,relative_closeness_coefficient,alternative_ranks):
        conn = psycopg2.connect(database=self.dbname, user=self.user, password=self.password,
                                host=self.host, port=self.port)
        cur = conn.cursor()
        
        
        cur.execute("""
                INSERT INTO results(results_sensitivity,weighted_normalized_decision_matrix,a_star,a_minus,positive_distance,negative_distance,relative_closeness_coefficient,alternative_ranks)
                VALUES (%s, %s,%s, %s,%s, %s,%s, %s)
            """, (json.dumps(results_sensitivity),json.dumps(weighted_normalized_decision_matrix),json.dumps(A_star),json.dumps(A_minus),json.dumps(positive_distance),json.dumps(negative_distance),json.dumps(relative_closeness_coefficient),json.dumps(alternative_ranks)))
        
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
        last_id = all_ids[-1][0]  
        print("Last ID in get_id_project:", last_id)
    else:
        last_id = None
    
    conn.close()
    return last_id



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
       
        session['criteria_names'] = criteria_names
        session['num_subcriteria'] = num_subcriteria
      
        return redirect(url_for('compare_criteria'))
    return render_template('add_criteria.html', num_criteria=num_criteria)




@app.route('/compare_subcriteria', methods=['GET', 'POST'])
def compare_subcriteria():
    criteria_names = session.get('criteria_names', [])
    num_subcriteria = session.get('num_subcriteria', [])
    weights_list = []
    rankings_list = []
    CR_list = []
    normalized_subcriteria_data = []  

    if request.method == 'POST':
        form_data = dict(request.form)
        db_manager = DatabaseManager("ahp_topsys_resultat", "postgres", "1234")
        print("FORM dataaaaaaaaa",form_data)

        criteria_data = []
        for i, criterion_name in enumerate(criteria_names):
           
            matrix = np.ones((num_subcriteria[i], num_subcriteria[i]))
            subcriteria_names = form_data.get(f'subcriteria_names_{i}').split('-')  

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

          
            criteria = Criteria(criterion_name)
            criteria.matrix = matrix  
            weights = criteria.calculate_weights(matrix)
            rankings = criteria.calculate_rankings(weights)
            criteria.check_consistency()
            CR = criteria.CR
            
           
            normalized_subcriteria_weights = criteria.calculate_normalized_subcriteria_weights(weights)
            print("hahaha",normalized_subcriteria_weights.tolist())
           
            weights_list.append(weights.tolist())
            rankings_list.append(rankings.tolist())
            CR_list.append(CR)
            
            criterion_data = (
                criterion_name, 
                subcriteria_names, 
                matrix.tolist(), 
                weights, 
                criteria.consistent,  
                CR,  
                rankings, 
                normalized_subcriteria_weights.tolist() 
            )
            criteria_data.append(criterion_data)

        db_manager.insert_criteria_data(criteria_data)

        return redirect(url_for('sub_cri_results'))
        

    return render_template('compare_subcriteria.html', criteria_names=criteria_names, num_subcriteria=num_subcriteria, normalized_subcriteria_data=normalized_subcriteria_data)

@app.route('/enter_comparison', methods=['GET', 'POST'])
def enter_comparison():
    if request.method == 'POST':
      
        return redirect(url_for('compare_subcriteria'))
    else:
   
        criteria_names = [...] 
        return render_template('enter_comparison.html', criteria_names=criteria_names)

@app.route('/compare_criteria', methods=['GET','POST'])
def compare_criteria():
    global global_num_criteria
    global nom_pro
    print("compare_criteria route accessed") 
    criteria_names = session.get('criteria_names', [])
    num_subcriteria = session.get('num_subcriteria', [])
    

    if request.method == 'POST':
        print("I'm inside post request")
        form_data = dict(request.form)
        product = Product()
        

        print("Form data", form_data)
        product.add_criteria(criteria_names)

        total_weights = []
        results = get_last_inserted_result()
        result_criterias = get_last_inserted_result_criteria()

        for criteria_tuple in result_criterias:
            total_weight = sum(criteria_tuple[3])  
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
        product.perform_topsis()  
        print("product.topsys",product.topsis_ranking)
        print("product.topsys.ideal",product.ideal_solution)
        print("product.topsys.negative",product.negative_ideal_solution)
        print("product.topsys.relative",product.relative_closeness)
        
       
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
    
    for criteria_tuple in result_criterias:
        total_weight = sum(criteria_tuple[3])  
        total_weights.append(total_weight)
    
    print("result_criterias", result_criterias)
    print("Total Weights:", total_weights)

    return render_template('calculate_results.html', results=results, result_criterias=result_criterias, total_weights=total_weights)   

def get_last_inserted_result():
    conn = psycopg2.connect(database="ahp_topsys_resultat", user="postgres",
                            password="1234", host="localhost", port="5432")
    cur = conn.cursor()
    id_pro=get_id_project(nom_pro)
    cur.execute("SELECT criteria, comparisons, weights, consistent, consistency_ratio, ranking,topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness FROM Product WHERE project_id = %s",(id_pro,))
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
weight_dic = dict
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
 

@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
    
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
 
@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Check if user is loggedin
    if 'loggedin' in session:
        return redirect(url_for('home'))

        
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
   
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
 
       
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
 
        if account:
            password_rs = account['password']
            print(password_rs)
            if check_password_hash(password_rs, password):
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                return redirect(url_for('home'))
            else:
                flash('Incorrect username/password')
        else:
            flash('Incorrect username/password')
 
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'loggedin' in session:
        return redirect(url_for('home'))
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    success = False

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        _hashed_password = generate_password_hash(password)

        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!')
        elif not username or not password or not email:
            flash('Please fill out the form!')
        else:
            cursor.execute("INSERT INTO users (fullname, username, password, email) VALUES (%s,%s,%s,%s)", (fullname, username, _hashed_password, email))
            conn.commit()
            flash('You have successfully registered!')
            success = True
    elif request.method == 'POST':
        flash('Please fill out the form!')

    return render_template('register.html', success=success)

   
@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))
  
@app.route('/profile')
def profile(): 
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM users WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))


@app.route('/topsyss', methods=['GET', 'POST'])
def show_topsis_form():
    if request.method == 'POST':
      
        total_weights = []

        results = get_last_inserted_result()
        result_criterias = get_last_inserted_result_criteria()

        print("*******route topsyss product",results)
        print("******route topsyss criteria",result_criterias)
        for criteria_tuple in result_criterias:
            total_weight = sum(criteria_tuple[3])  
            total_weights.append(total_weight)


        num_alternatives = int(request.form.get('numAlternatives', 0))
        alternatives = [request.form.get(f'alternative{i}') for i in range(num_alternatives)]

        criteria_weights = {f'Criterion {i + 1}': round(np.random.rand(), 2) for i in
                            range(3)} 

        return render_template('topsyss.html', alternatives=alternatives, criteria_weights=criteria_weights,
                               results=results, result_criterias=result_criterias, total_weights=total_weights
        )

    return render_template('topsyss.html', alternatives=[], criteria_weights={})
alternative_noms = {}
criteria_weights_test ={}
alternative_values_test ={}
@app.route('/process_topsis', methods=['POST'])
def process_topsis():
    global weight_dic 
    global alternative_noms
    global results_of_exchanging
    global criteria_weights_test
    global alternative_values_test
    form_data = dict(request.form)
    print("FROM in process_topsis", form_data)
    num_alternatives = int(request.form.get('numAlternatives', 0))
    alternatives = [request.form.get(f'alternative{i}') for i in range(num_alternatives)]
    criteria_weights = {}
    alternative_values = {}
    
    for key, value in form_data.items():
        if key.startswith('subcriterion_weight_'):
            subcriterion_name = key.split('_')[-1]
            weight = float(value)
            criteria_weights[subcriterion_name] = weight
        elif key.startswith('alternative_values_'):
            parts = key.split('_')
            subcriterion_name = '_'.join(parts[2:-1])  
            alternative_name = parts[-1]  
            value = float(value)
            if subcriterion_name not in alternative_values:
                alternative_values[subcriterion_name] = {}
            alternative_values[subcriterion_name][alternative_name] = value
    
    print("criteria_weights",criteria_weights)
    print("alternative_values",alternative_values)
    alternative_values_array = np.array([[alternative_values[criterion][alternative] for alternative in alternative_values[criterion]] for criterion in criteria_weights])

    

    normalized_alternatives = alternative_values_array / np.sqrt((alternative_values_array ** 2).sum(axis=1))[:, np.newaxis]

    criteria_weights_array = np.array(list(criteria_weights.values()))

    weighted_normalized_decision_matrix = normalized_alternatives * criteria_weights_array[:, np.newaxis]
    
    result_criterias = get_last_inserted_result_criteria()
    print("Weighted Normalized Decision Matrix:")
    print(weighted_normalized_decision_matrix)
    subcriteria_list = list(alternative_values.keys())
    weighted_normalized_decision_matrix_dict = OrderedDict()
    weighted_normalized_decision_matrix_dict2 = OrderedDict()
    for i, subcriterion in enumerate(subcriteria_list):
        weighted_normalized_decision_matrix_dict[subcriterion] = weighted_normalized_decision_matrix[i]

    alternative_names = {}
    for subcriterion_name, values in alternative_values.items():
        for alternative_name in values:
            if alternative_name not in alternative_names:
                alternative_names[alternative_name] = []
            alternative_names[alternative_name].append(subcriterion_name)

    criteria_weights_test=criteria_weights
    alternative_values_test=alternative_values
    
    
    print("Weighted Normalized Decision Matrix diccccc:",weighted_normalized_decision_matrix_dict)
    print("alternative_names",alternative_names)
    weight_dic = weighted_normalized_decision_matrix_dict 
    alternative_noms = alternative_names
    return render_template('results.html', criteria_weights=criteria_weights, 
                           weighted_normalized_decision_matrix=weighted_normalized_decision_matrix_dict, 
                           alternative_names=alternative_names)

results_sensitivity_test = []
def echange_weights(criteria_weights,alternative_values):
    global results_sensitivity_test 
    subcriteria = list(criteria_weights.keys())

    for criterion in alternative_values.keys():
        alternative_values_array = np.array([list(alternative_values[criterion].values())])
        normalized_values = alternative_values_array / np.sqrt((alternative_values_array ** 2).sum(axis=1))[:, np.newaxis]
        alternative_values[criterion] = normalized_values

    criteria_weights_array = np.array(list(criteria_weights.values()))
    weighted_normalized_decision_matrix = np.array([alternative_values[subcriterion] * weight for subcriterion, weight in zip(subcriteria, criteria_weights_array)])

    results_sensitivity_test.append({
        "criteria_exchange": ("Constant", "No Change"),
        "criteria_weights": criteria_weights.copy(),  
        "weighted_normalized_decision_matrix": weighted_normalized_decision_matrix
    })

    for j in range(1, len(subcriteria)):
        modified_weights = criteria_weights.copy()
        
        modified_weights[subcriteria[0]] = criteria_weights[subcriteria[j]]
        modified_weights[subcriteria[j]] = criteria_weights[subcriteria[0]]

        criteria_weights_array = np.array(list(modified_weights.values()))

        weighted_normalized_decision_matrix = np.array([alternative_values[subcriterion] * weight for subcriterion, weight in zip(subcriteria, criteria_weights_array)])

        case_results = {
            "criteria_exchange": (subcriteria[0], subcriteria[j]),
            "criteria_weights": modified_weights,
            "weighted_normalized_decision_matrix": weighted_normalized_decision_matrix
        }
        
        results_sensitivity_test.append(case_results)

    equal_weight = {subcriterion: round(1 / len(subcriteria), 4) for subcriterion in subcriteria}
    results_sensitivity_test.append({
        "criteria_exchange": ("Equal", "Weights"),
        "criteria_weights": equal_weight
    })


    return results_sensitivity_test

criteria_weights_sen = {}

results_sensitivity_sen ={}

exchange_data = []

import math
exchange_weight_final_resuts = []



@app.route('/negative_postive_alter', methods=['POST'])
def negative_postive_alter():
    # Get the form data
    global alternative_noms
    global weight_dic
    global criteria_weights_sen 
    global results_sensitivity_sen 
    global alternative_ranks2
    global exchange_weight_final_resuts
    global exchange_data 
    criteria_weights = {}
    weighted_normalized_decision_matrix = defaultdict(list)
    maximize_minimize2 = {}
    alternative_values = defaultdict(dict)

    for key, value in request.form.items():
        if key.endswith('_weight'):
            subcriterion = key.split('_')[0]
            criteria_weights[subcriterion] = float(value)
        elif key.endswith('_value'):
            parts = key.split('_')
            subcriterion = parts[0]
            alternative_name = parts[1]
            alternative_values[alternative_name][subcriterion] = float(value)
            weighted_normalized_decision_matrix[subcriterion].append(float(value))
        elif key.startswith('maximize_') or key.startswith('minimize_'):
            subcriterion = key.split('_')[1]
            maximize_minimize2[subcriterion] = value
    
    A_star = {}
    A_minus = {}

    for subcriterion, values in weighted_normalized_decision_matrix.items():
        if len(values) == 0:
            print("Values list is empty for sub-criterion:", subcriterion)
            continue

        if maximize_minimize2[subcriterion] == 'maximize':
            A_star[subcriterion] = max(values)
            A_minus[subcriterion] = min(values)
            if A_minus[subcriterion] > A_star[subcriterion]:
                A_star[subcriterion], A_minus[subcriterion] = A_minus[subcriterion], A_star[subcriterion]
        else:
            A_star[subcriterion] = min(values)
            A_minus[subcriterion] = max(values)

    positive_distances = {}
    negative_distances = {}

    for alternative, values in alternative_values.items():
        positive_distance = math.sqrt(sum((values[criterion] - A_star[criterion])**2 for criterion in values))
        positive_distances[alternative] = positive_distance

        negative_distance = math.sqrt(sum((values[criterion] - A_minus[criterion])**2 for criterion in values))
        negative_distances[alternative] = negative_distance

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

        relative_closeness_coefficient = negative_distance / (positive_distance + negative_distance)
        
        relative_closeness_coefficients[alternative] = relative_closeness_coefficient

    print("Relative Closeness Coefficients:", relative_closeness_coefficients)
    ranked_alternatives = sorted(relative_closeness_coefficients.items(), key=lambda x: x[1], reverse=True)
    print("Ranked Alternatives ",ranked_alternatives)
    optimal_alternative, optimal_closeness_coefficient = ranked_alternatives[0]
    alternative_ranks = {}

    print("Ranked Alternatives (Descending Order):")
    for rank, (alternative, closeness_coefficient) in enumerate(ranked_alternatives, start=1):
        print("Rank", rank, "-", alternative, ": Closeness Coefficient =", closeness_coefficient)
        alternative_ranks[alternative] = rank


   
    optimal_alternative, optimal_closeness_coefficient = ranked_alternatives[0]
    print("\nOptimal Alternative:", optimal_alternative)
    print("Alternative Ranks:", alternative_ranks)
    results_sensitivity = []
    subcriteria = list(criteria_weights.keys())
    print("length of subcriteria:", len(subcriteria))
    results_sensitivity.append({
        "criteria_exchange": ("Constant", "No Change"),
        "criteria_weights": criteria_weights.copy()  
    })

    for j in range(1, len(subcriteria)):
        modified_weights = dict(criteria_weights)
        
        modified_weights[subcriteria[0]], modified_weights[subcriteria[j]] = modified_weights[subcriteria[j]], modified_weights[subcriteria[0]]
        print("{", modified_weights[subcriteria[0]], ",", modified_weights[subcriteria[j]], "}")
        case_results = {
            "criteria_exchange": (subcriteria[0], subcriteria[j]),
            "criteria_weights": modified_weights
        }
        

        results_sensitivity.append(case_results)

    

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


        
    print("*****************************************************")
    print("results_sensitivity:", results_sensitivity)
  

    results_of_exchanging=echange_weights(criteria_weights_test, alternative_values_test)

    A_Star2 = {}
    A_Min2 = {}

    
    def calculate_A_Star2_A_Min2(matrix_section, maximize_minimize):
        A_Star2 = np.max(matrix_section) if maximize_minimize == 'maximize' else np.min(matrix_section)
        A_Min2 = np.min(matrix_section) if maximize_minimize == 'maximize' else np.max(matrix_section)
        return A_Star2, A_Min2

    results = []

    for exchange_result in results_of_exchanging:
        exchange = exchange_result['criteria_exchange']
        if exchange[0] != 'Equal': 
            weighted_matrix = exchange_result['weighted_normalized_decision_matrix']
            section_results_A_Star2 = {}
            section_results_A_Min2 = {}
            print("weighted_matrix",weighted_matrix)
            for i, criteria in enumerate(maximize_minimize2):
                A_Star2, A_Min2 = calculate_A_Star2_A_Min2(weighted_matrix[i][0], maximize_minimize2[criteria])
                
                section_results_A_Star2[criteria] = A_Star2
                section_results_A_Min2[criteria] = A_Min2
            
            exchange_result['A_Star2'] = section_results_A_Star2
            exchange_result['A_Min2'] = section_results_A_Min2

    

   

    for exchange_result in results_of_exchanging:
        exchange = exchange_result['criteria_exchange']
        if exchange[0] != 'Equal':  
            weighted_matrix = exchange_result['weighted_normalized_decision_matrix']
            A_Star2 = exchange_result['A_Star2']
            A_Min2 = exchange_result['A_Min2']
            positive_dis2 = {}
            negative_dis2 = {}

            criterion_names = list(A_Star2.keys())
            criterion_names2 = list(A_Min2.keys())
            
            all_weights = []

            for alternative_weights in weighted_matrix:
                alternative_weights_list = []
                for weight_list in alternative_weights:
                    alternative_weights_list.extend(weight_list)
                all_weights.append(alternative_weights_list)

            alternatives_weights = list(map(list, zip(*all_weights)))
            print("alternatives_weights",alternatives_weights)
            for i, weights in enumerate(alternatives_weights):
                alternative = f'Alternative {i+1}' 
                positive_distance = math.sqrt(sum((weights[k] - A_Star2[criterion_names[k]]) ** 2 for k in range(len(weights))))
                negative_distance = math.sqrt(sum((weights[k] - A_Min2[criterion_names2[k]]) ** 2 for k in range(len(weights))))
                positive_dis2[alternative] = positive_distance
                negative_dis2[alternative] = negative_distance

            exchange_result['positive_dis2'] = positive_dis2
            exchange_result['negative_dis2'] = negative_dis2


    
    exchange_data = []

    for exchange_result in results_of_exchanging:
        exchange = exchange_result['criteria_exchange']
        if exchange[0] != 'Equal': 
            weighted_matrix = exchange_result['weighted_normalized_decision_matrix']
            A_Star2 = exchange_result['A_Star2']
            A_Min2 = exchange_result['A_Min2']
            positive_dis2 = exchange_result.get('positive_dis2', {})
            negative_dis2 = exchange_result.get('negative_dis2', {})
            
            criterion_names = list(A_Star2.keys())
            
            Ci_values = {}
            
            for i, weights in enumerate(alternatives_weights):
                alternative = f'Alternative {i+1}'  
                positive_distance = positive_dis2.get(alternative, 0) 
                negative_distance = negative_dis2.get(alternative, 0)  
                
                Ci = negative_distance / (positive_distance + negative_distance)
                Ci_values[alternative] = Ci
            
            exchange_result['Ci_values'] = Ci_values
            
            ranked_alternatives = sorted(Ci_values.items(), key=lambda x: x[1], reverse=True)
            
            ranked_list = [alternative for alternative, _ in ranked_alternatives]
            
            exchange_result['ranked_alternatives'] = ranked_list
            ci_data ={}
            ci_data = {
                'Ci_values': Ci_values,
                'ranked_alternatives': ranked_list
            }
            exchange_data.append(ci_data)

    print("exchange_data:", exchange_data)
            
    print("results_of_exchanging_final",results_of_exchanging)

    exchange_weight_final_resuts = results_of_exchanging
    

    criteria_weights_sen = criteria_weights 
    results_sensitivity_sen = results_sensitivity
    alternative_noms=alternative_values
    alternative_ranks2=alternative_ranks
    db_manager = DatabaseManager("ahp_topsys_resultat", "postgres", "1234")
    db_manager.insert_alternatives_and_sensitivity(alternative_values,results_sensitivity)
    db_manager.insert_results(results_sensitivity,weighted_normalized_decision_matrix,A_star,A_minus,positive_distance,negative_distance,relative_closeness_coefficient,alternative_ranks)
 
    

    
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
    global exchange_weight_final_resuts
    global exchange_data 
    print("graph route")
    
    print("exchange_weight_final_resuts",exchange_weight_final_resuts)
    print("exchange_data",exchange_data)
    return render_template('Graph.html',exchange_weight_final_resuts=exchange_weight_final_resuts,
     exchange_data=exchange_data,
     alternative_noms=alternative_noms)

import matplotlib.pyplot as plt
import numpy as np
from flask import send_file

@app.route('/designer_graph')
def designer_graph():
    global exchange_data 

    alternatives = list(exchange_data[0]['Ci_values'].keys())

    ranks = {alternative: [] for alternative in alternatives}

    for exchange in exchange_data:
        ranked_alternatives = exchange['ranked_alternatives']
        for rank, alternative in enumerate(ranked_alternatives, start=1):
            ranks[alternative].append(rank)

    for alternative, rank_list in ranks.items():
        plt.plot(range(1, len(rank_list) + 1), rank_list, marker='o', label=alternative)

    plt.xlabel('Exchange')
    plt.ylabel('Rank')
    plt.title('Rank of Alternatives Across Exchanges')
    plt.legend()

    
    plt.show()
    plt.close()
    

    return render_template('Final_Step.html')

@app.route('/sensitivity')
def sensitivity():
    print("sensitivity roooooooooooute ***********")
    print("criteria_weights_sen",criteria_weights_sen)
    print("results_sensitivity_sen",results_sensitivity_sen)


    return render_template('sensitivity.html',criteria_weights_sen=criteria_weights_sen,
    results_sensitivity_sen=results_sensitivity_sen)

@app.route('/get_subcriteria_names')
def get_subcriteria_names():
    subcriteria_names = get_subcriteria_names_from_database()  

    return jsonify({'subcriteria_names': subcriteria_names})

def get_projects_by_user(user_id):
    try:
       
        conn = psycopg2.connect(database="ahp_topsys_resultat", user="postgres", password="1234", host="localhost", port="5432")
        cur = conn.cursor()
        cur.execute("SELECT id_proj, name FROM project WHERE user_id = %s", (user_id,))
        projects = cur.fetchall()
        project_info = []
        

        for project_id, project_name in projects:
           
            cur.execute("SELECT criterion_name, subcriteria_data, subcriteria_comparisons, weights, consistent, consistency_ratio, ranking, normalized_subcriteria_weights FROM criteria WHERE project_id = %s", (project_id,))
            project_data = cur.fetchall()

            cur.execute("SELECT criteria, comparisons, weights, consistent, consistency_ratio, ranking,topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness FROM Product WHERE project_id = %s",(project_id,))
            result = cur.fetchall()

            project_info.append({
                'id': project_id,
                'name': project_name,
                'data_criteria':result,
                'data': project_data
            })

        cur.close()
        conn.close()
        return project_info

    except (Exception, psycopg2.Error) as error:
        print("Error fetching projects:", error)
        return []

@app.route('/history')
def history():
    id_user =session['id']
    project_informations = get_projects_by_user(id_user)
    print("project_informations",project_informations)
    return render_template('history.html',project_informations=project_informations)

if __name__ == '__main__':
    app.run(debug=True)

