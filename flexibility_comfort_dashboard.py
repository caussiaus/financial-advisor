#!/usr/bin/env python3
"""
Interactive Flexibility vs. Comfort Financial Mesh Dashboard

A web application that allows users to:
1. Input client financial data
2. Visualize the financial mesh from flexibility vs. comfort perspective
3. See real-time recommendations
4. Compare planned vs. actual scenarios
5. Adjust payment structures using PV calculations
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, jsonify, send_file
import json
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
import plotly.express as px

from src.layers.flexibility_comfort_mesh import (
    FlexibilityComfortMeshEngine, ClientProfile, PaymentType
)


app = Flask(__name__)

# Initialize the mesh engine
mesh_engine = FlexibilityComfortMeshEngine()

# Sample client data (can be modified through UI)
sample_clients = {
    "john_smith": {
        "name": "John Smith",
        "age": 45,
        "current_assets": {
            "cash": 150000,
            "stocks": 300000,
            "bonds": 200000,
            "real_estate": 100000
        },
        "income": 120000,
        "expenses": 80000,
        "risk_tolerance": "moderate",
        "time_horizon": 20,
        "financial_goals": [
            {"name": "College Education", "amount": 50000, "year": 2025},
            {"name": "Home Renovation", "amount": 75000, "year": 2024},
            {"name": "Retirement", "amount": 2000000, "year": 2035}
        ],
        "payment_preferences": {
            "college_education": PaymentType.PV_CALCULATED,
            "home_renovation": PaymentType.PLANNED,
            "retirement": PaymentType.ANNUITY
        }
    },
    "sarah_jones": {
        "name": "Sarah Jones",
        "age": 35,
        "current_assets": {
            "cash": 80000,
            "stocks": 250000,
            "bonds": 150000,
            "real_estate": 200000
        },
        "income": 95000,
        "expenses": 65000,
        "risk_tolerance": "conservative",
        "time_horizon": 25,
        "financial_goals": [
            {"name": "Emergency Fund", "amount": 50000, "year": 2024},
            {"name": "Investment Property", "amount": 300000, "year": 2028},
            {"name": "Retirement", "amount": 1500000, "year": 2040}
        ],
        "payment_preferences": {
            "emergency_fund": PaymentType.LUMP_SUM,
            "investment_property": PaymentType.PV_CALCULATED,
            "retirement": PaymentType.ANNUITY
        }
    }
}


def create_client_profile(client_data):
    """Create a ClientProfile from dictionary data"""
    return ClientProfile(
        name=client_data["name"],
        age=client_data["age"],
        current_assets=client_data["current_assets"],
        income=client_data["income"],
        expenses=client_data["expenses"],
        risk_tolerance=client_data["risk_tolerance"],
        time_horizon=client_data["time_horizon"],
        financial_goals=client_data["financial_goals"],
        payment_preferences=client_data["payment_preferences"]
    )


def calculate_pv_payment(goal_amount, years_to_goal, discount_rate=0.05):
    """Calculate present value payment for a future goal"""
    if years_to_goal <= 0:
        return goal_amount
    
    pv_factor = 1 / ((1 + discount_rate) ** years_to_goal)
    return goal_amount * pv_factor


def calculate_annuity_payment(goal_amount, years_to_goal, interest_rate=0.05):
    """Calculate annuity payment for a future goal"""
    if years_to_goal <= 0:
        return goal_amount
    
    # Annuity payment formula: PMT = FV * r / ((1 + r)^n - 1)
    r = interest_rate
    n = years_to_goal
    annuity_factor = r / ((1 + r) ** n - 1)
    return goal_amount * annuity_factor


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', clients=sample_clients)


@app.route('/api/client/<client_id>')
def get_client_data(client_id):
    """Get client data"""
    if client_id in sample_clients:
        return jsonify(sample_clients[client_id])
    return jsonify({"error": "Client not found"}), 404


@app.route('/api/client/<client_id>', methods=['POST'])
def update_client_data(client_id):
    """Update client data"""
    data = request.json
    
    if client_id not in sample_clients:
        return jsonify({"error": "Client not found"}), 404
    
    # Update client data
    sample_clients[client_id].update(data)
    
    # Recreate mesh visualization
    client_profile = create_client_profile(sample_clients[client_id])
    mesh_engine.add_client_profile(client_profile)
    
    return jsonify({"success": True, "message": "Client data updated"})


@app.route('/api/mesh/<client_id>')
def get_mesh_data(client_id):
    """Get mesh visualization data for a client"""
    if client_id not in sample_clients:
        return jsonify({"error": "Client not found"}), 404
    
    client_data = sample_clients[client_id]
    client_profile = create_client_profile(client_data)
    mesh_engine.add_client_profile(client_profile)
    
    # Generate scenarios and visualization
    scenarios = mesh_engine.generate_scenarios(client_profile.name, num_scenarios=500)
    viz = mesh_engine.create_mesh_visualization(client_profile.name)
    recommendations = mesh_engine.generate_recommendations(client_profile.name)
    
    # Calculate current position
    avg_flexibility = np.mean(viz.x_coords)
    avg_comfort = np.mean(viz.y_coords)
    
    # Calculate goal payments
    current_year = datetime.now().year
    goal_payments = []
    
    for goal in client_data["financial_goals"]:
        years_to_goal = goal["year"] - current_year
        goal_amount = goal["amount"]
        goal_name = goal["name"].lower().replace(" ", "_")
        
        payment_type = client_data["payment_preferences"].get(goal_name, PaymentType.PLANNED)
        
        if payment_type == PaymentType.PV_CALCULATED:
            payment = calculate_pv_payment(goal_amount, years_to_goal)
            payment_type_str = "PV Calculated"
        elif payment_type == PaymentType.ANNUITY:
            payment = calculate_annuity_payment(goal_amount, years_to_goal)
            payment_type_str = "Annuity"
        elif payment_type == PaymentType.LUMP_SUM:
            payment = goal_amount
            payment_type_str = "Lump Sum"
        else:  # PLANNED
            payment = goal_amount / max(years_to_goal, 1)
            payment_type_str = "Planned"
        
        goal_payments.append({
            "name": goal["name"],
            "amount": goal_amount,
            "year": goal["year"],
            "years_to_goal": years_to_goal,
            "payment_type": payment_type_str,
            "payment_amount": payment
        })
    
    mesh_data = {
        "scenarios": [
            {
                "x": float(x),
                "y": float(y),
                "color": color,
                "size": float(size),
                "scenario_id": scenario_id
            }
            for x, y, color, size, scenario_id in zip(
                viz.x_coords, viz.y_coords, viz.colors, viz.sizes, viz.scenario_ids
            )
        ],
        "target_zone": viz.target_zone,
        "optimal_path": viz.optimal_path,
        "current_position": {
            "flexibility": float(avg_flexibility),
            "comfort": float(avg_comfort)
        },
        "recommendations": recommendations,
        "goal_payments": goal_payments,
        "client_summary": {
            "total_assets": sum(client_data["current_assets"].values()),
            "income": client_data["income"],
            "expenses": client_data["expenses"],
            "savings_rate": (client_data["income"] - client_data["expenses"]) / client_data["income"]
        }
    }
    
    return jsonify(mesh_data)


@app.route('/api/compare/<client_id>')
def compare_planned_vs_actual(client_id):
    """Compare planned vs actual scenarios"""
    if client_id not in sample_clients:
        return jsonify({"error": "Client not found"}), 404
    
    client_data = sample_clients[client_id]
    
    # Generate planned scenario (current allocation)
    planned_assets = client_data["current_assets"]
    planned_income = client_data["income"]
    planned_expenses = client_data["expenses"]
    
    # Generate actual scenario (with some variation)
    actual_assets = {}
    for asset, amount in planned_assets.items():
        # Add some random variation
        variation = np.random.normal(1.0, 0.1)
        actual_assets[asset] = amount * variation
    
    actual_income = planned_income * np.random.normal(1.0, 0.05)
    actual_expenses = planned_expenses * np.random.normal(1.0, 0.08)
    
    # Calculate metrics for both scenarios
    planned_risk = mesh_engine.compute_risk_metrics(planned_assets)
    actual_risk = mesh_engine.compute_risk_metrics(actual_assets)
    
    planned_flexibility = mesh_engine.compute_flexibility_score(planned_assets, planned_income, planned_expenses)
    actual_flexibility = mesh_engine.compute_flexibility_score(actual_assets, actual_income, actual_expenses)
    
    planned_comfort = mesh_engine.compute_comfort_score(planned_assets, planned_risk, 1.0 - planned_risk['volatility'])
    actual_comfort = mesh_engine.compute_comfort_score(actual_assets, actual_risk, 1.0 - actual_risk['volatility'])
    
    comparison_data = {
        "planned": {
            "assets": planned_assets,
            "income": planned_income,
            "expenses": planned_expenses,
            "flexibility": planned_flexibility,
            "comfort": planned_comfort,
            "risk_metrics": planned_risk
        },
        "actual": {
            "assets": actual_assets,
            "income": actual_income,
            "expenses": actual_expenses,
            "flexibility": actual_flexibility,
            "comfort": actual_comfort,
            "risk_metrics": actual_risk
        },
        "differences": {
            "flexibility_change": actual_flexibility - planned_flexibility,
            "comfort_change": actual_comfort - planned_comfort,
            "total_assets_change": sum(actual_assets.values()) - sum(planned_assets.values()),
            "savings_rate_change": (actual_income - actual_expenses) / actual_income - (planned_income - planned_expenses) / planned_income
        }
    }
    
    return jsonify(comparison_data)


@app.route('/api/payment_calculator', methods=['POST'])
def calculate_payment():
    """Calculate payment for a financial goal"""
    data = request.json
    
    goal_amount = data.get('amount', 0)
    years_to_goal = data.get('years', 0)
    payment_type = data.get('payment_type', 'planned')
    interest_rate = data.get('interest_rate', 0.05)
    
    if payment_type == 'pv_calculated':
        payment = calculate_pv_payment(goal_amount, years_to_goal, interest_rate)
    elif payment_type == 'annuity':
        payment = calculate_annuity_payment(goal_amount, years_to_goal, interest_rate)
    elif payment_type == 'lump_sum':
        payment = goal_amount
    else:  # planned
        payment = goal_amount / max(years_to_goal, 1)
    
    return jsonify({
        "payment_amount": payment,
        "payment_type": payment_type,
        "goal_amount": goal_amount,
        "years_to_goal": years_to_goal
    })


@app.route('/dashboard/<client_id>')
def client_dashboard(client_id):
    """Individual client dashboard page"""
    if client_id not in sample_clients:
        return "Client not found", 404
    
    return render_template('client_dashboard.html', 
                         client_id=client_id, 
                         client_data=sample_clients[client_id])


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create dashboard template
    dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Flexibility vs. Comfort Financial Mesh Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .panel {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #e9ecef;
        }
        .client-selector {
            margin-bottom: 20px;
        }
        .client-selector select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .mesh-chart {
            height: 500px;
            background: white;
            border-radius: 8px;
            padding: 10px;
        }
        .recommendations {
            max-height: 500px;
            overflow-y: auto;
        }
        .recommendation {
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        .recommendation.high {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        .recommendation.medium {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        .recommendation.low {
            background-color: #e8f5e8;
            border-left-color: #4caf50;
        }
        .client-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Flexibility vs. Comfort Financial Mesh Dashboard</h1>
            <p>Visualize your financial future through the lens of flexibility and comfort</p>
        </div>
        
        <div class="content">
            <div class="panel">
                <div class="client-selector">
                    <label for="clientSelect">Select Client:</label>
                    <select id="clientSelect">
                        <option value="">Choose a client...</option>
                        {% for client_id, client_data in clients.items() %}
                        <option value="{{ client_id }}">{{ client_data.name }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="metrics" id="metrics" style="display: none;">
                    <div class="metric-card">
                        <div class="metric-value" id="totalAssets">$0</div>
                        <div class="metric-label">Total Assets</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="flexibilityScore">0.0</div>
                        <div class="metric-label">Flexibility Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="comfortScore">0.0</div>
                        <div class="metric-label">Comfort Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="savingsRate">0%</div>
                        <div class="metric-label">Savings Rate</div>
                    </div>
                </div>
                
                <div class="mesh-chart" id="meshChart"></div>
            </div>
            
            <div class="panel">
                <h3>Recommendations</h3>
                <div class="recommendations" id="recommendations"></div>
                
                <h3>Financial Goals & Payments</h3>
                <div id="goalPayments"></div>
                
                <h3>Client Data Editor</h3>
                <div class="client-form" id="clientForm" style="display: none;">
                    <div class="form-group">
                        <label>Name:</label>
                        <input type="text" id="clientName" />
                    </div>
                    <div class="form-group">
                        <label>Age:</label>
                        <input type="number" id="clientAge" />
                    </div>
                    <div class="form-group">
                        <label>Income:</label>
                        <input type="number" id="clientIncome" />
                    </div>
                    <div class="form-group">
                        <label>Expenses:</label>
                        <input type="number" id="clientExpenses" />
                    </div>
                    <div class="form-group">
                        <label>Risk Tolerance:</label>
                        <select id="clientRiskTolerance">
                            <option value="conservative">Conservative</option>
                            <option value="moderate">Moderate</option>
                            <option value="aggressive">Aggressive</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Cash:</label>
                        <input type="number" id="clientCash" />
                    </div>
                    <div class="form-group">
                        <label>Stocks:</label>
                        <input type="number" id="clientStocks" />
                    </div>
                    <div class="form-group">
                        <label>Bonds:</label>
                        <input type="number" id="clientBonds" />
                    </div>
                    <div class="form-group">
                        <label>Real Estate:</label>
                        <input type="number" id="clientRealEstate" />
                    </div>
                    <button class="btn" onclick="updateClientData()">Update Client Data</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentClientId = null;
        
        $('#clientSelect').change(function() {
            const clientId = $(this).val();
            if (clientId) {
                loadClientData(clientId);
            }
        });
        
        function loadClientData(clientId) {
            currentClientId = clientId;
            
            // Load mesh data
            $.get(`/api/mesh/${clientId}`)
                .done(function(data) {
                    updateMeshChart(data);
                    updateRecommendations(data.recommendations);
                    updateGoalPayments(data.goal_payments);
                    updateMetrics(data.client_summary, data.current_position);
                    loadClientForm(clientId);
                })
                .fail(function(xhr, status, error) {
                    console.error('Error loading client data:', error);
                });
        }
        
        function updateMeshChart(data) {
            const scenarios = data.scenarios;
            
            const trace1 = {
                x: scenarios.map(s => s.x),
                y: scenarios.map(s => s.y),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: scenarios.map(s => s.size),
                    color: scenarios.map(s => s.color),
                    opacity: 0.7
                },
                text: scenarios.map((s, i) => `Scenario ${i + 1}`),
                hovertemplate: '<b>%{text}</b><br>Flexibility: %{x:.3f}<br>Comfort: %{y:.3f}<extra></extra>',
                name: 'Financial Scenarios'
            };
            
            const trace2 = {
                x: data.target_zone.map(p => p[0]) + [data.target_zone[0][0]],
                y: data.target_zone.map(p => p[1]) + [data.target_zone[0][1]],
                mode: 'lines',
                type: 'scatter',
                line: {color: 'green', width: 3, dash: 'dash'},
                name: 'Target Zone'
            };
            
            const trace3 = {
                x: data.optimal_path.map(p => p[0]),
                y: data.optimal_path.map(p => p[1]),
                mode: 'lines+markers',
                type: 'scatter',
                line: {color: 'blue', width: 2},
                marker: {size: 8, color: 'blue'},
                name: 'Optimal Path'
            };
            
            const layout = {
                title: 'Financial Mesh: Flexibility vs. Comfort',
                xaxis: {title: 'Flexibility Score', range: [0, 1]},
                yaxis: {title: 'Comfort Score', range: [0, 1]},
                width: 600,
                height: 500
            };
            
            Plotly.newPlot('meshChart', [trace1, trace2, trace3], layout);
        }
        
        function updateRecommendations(recommendations) {
            const container = $('#recommendations');
            container.empty();
            
            recommendations.forEach(rec => {
                const recHtml = `
                    <div class="recommendation ${rec.priority}">
                        <h4>${rec.type.charAt(0).toUpperCase() + rec.type.slice(1)}</h4>
                        <p>${rec.description}</p>
                        <ul>
                            ${rec.actions.map(action => `<li>${action}</li>`).join('')}
                        </ul>
                    </div>
                `;
                container.append(recHtml);
            });
        }
        
        function updateGoalPayments(goalPayments) {
            const container = $('#goalPayments');
            container.empty();
            
            goalPayments.forEach(goal => {
                const goalHtml = `
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <h4>${goal.name}</h4>
                        <p><strong>Amount:</strong> $${goal.amount.toLocaleString()}</p>
                        <p><strong>Year:</strong> ${goal.year} (${goal.years_to_goal} years)</p>
                        <p><strong>Payment Type:</strong> ${goal.payment_type}</p>
                        <p><strong>Payment Amount:</strong> $${goal.payment_amount.toLocaleString()}</p>
                    </div>
                `;
                container.append(goalHtml);
            });
        }
        
        function updateMetrics(summary, position) {
            $('#totalAssets').text('$' + summary.total_assets.toLocaleString());
            $('#flexibilityScore').text(position.flexibility.toFixed(3));
            $('#comfortScore').text(position.comfort.toFixed(3));
            $('#savingsRate').text((summary.savings_rate * 100).toFixed(1) + '%');
            $('#metrics').show();
        }
        
        function loadClientForm(clientId) {
            $.get(`/api/client/${clientId}`)
                .done(function(data) {
                    $('#clientName').val(data.name);
                    $('#clientAge').val(data.age);
                    $('#clientIncome').val(data.income);
                    $('#clientExpenses').val(data.expenses);
                    $('#clientRiskTolerance').val(data.risk_tolerance);
                    $('#clientCash').val(data.current_assets.cash);
                    $('#clientStocks').val(data.current_assets.stocks);
                    $('#clientBonds').val(data.current_assets.bonds);
                    $('#clientRealEstate').val(data.current_assets.real_estate);
                    $('#clientForm').show();
                });
        }
        
        function updateClientData() {
            if (!currentClientId) return;
            
            const updatedData = {
                name: $('#clientName').val(),
                age: parseInt($('#clientAge').val()),
                income: parseFloat($('#clientIncome').val()),
                expenses: parseFloat($('#clientExpenses').val()),
                risk_tolerance: $('#clientRiskTolerance').val(),
                current_assets: {
                    cash: parseFloat($('#clientCash').val()),
                    stocks: parseFloat($('#clientStocks').val()),
                    bonds: parseFloat($('#clientBonds').val()),
                    real_estate: parseFloat($('#clientRealEstate').val())
                }
            };
            
            $.ajax({
                url: `/api/client/${currentClientId}`,
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(updatedData)
            })
            .done(function(response) {
                alert('Client data updated successfully!');
                loadClientData(currentClientId);
            })
            .fail(function(xhr, status, error) {
                alert('Error updating client data: ' + error);
            });
        }
    </script>
</body>
</html>
"""
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(dashboard_template)
    
    print("🚀 Starting Flexibility vs. Comfort Financial Mesh Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("💡 Features:")
    print("   - Interactive mesh visualization")
    print("   - Real-time client data editing")
    print("   - Flexibility vs. comfort analysis")
    print("   - Payment structure calculations")
    print("   - Planned vs. actual comparisons")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 