import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Compute first innings total scores
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning']==1]

# Merge with match info
match_df = match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')

# Remove discontinued teams
remove_teams = ['Deccan Chargers','Delhi Daredevils','Pune Warriors',
                'Kochi Tuskers Kerala','Gujarat Lions']
match_df = match_df[~match_df['team1'].isin(remove_teams)]
match_df = match_df[~match_df['team2'].isin(remove_teams)]

# Remove matches with method (D/L)
match_df = match_df[match_df['method'].isna()]

# Keep relevant columns
match_df = match_df[['id','city','winner','total_runs']]
match_df.rename(columns={'total_runs':'target_runs'}, inplace=True)
match_df['target_runs'] = match_df['target_runs'] + 1  # chasing target

# Merge with deliveries for 2nd innings
delivery_df = match_df.merge(delivery, left_on='id', right_on='match_id')
delivery_df = delivery_df[delivery_df['inning']==2]

# Calculate features
delivery_df['total_runs']=pd.to_numeric(delivery_df['total_runs'], errors='coerce')
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
delivery_df['runs_left'] = delivery_df['target_runs'] - delivery_df['current_score']
delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])
delivery_df['wicket'] = delivery_df['player_dismissed'].notna().astype(int)
delivery_df['wickets_fallen'] = delivery_df.groupby('match_id')['wicket'].cumsum()
delivery_df['wickets_left'] = 10 - delivery_df['wickets_fallen']
delivery_df['crr'] = (delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']
delivery_df['result'] = np.where(delivery_df['batting_team']==delivery_df['winner'],1,0)

# Select final features
final_df = delivery_df[['match_id','batting_team','bowling_team','city',
                        'runs_left','balls_left','wickets_left','total_runs','crr','rrr','result']]

# Remove rows with balls_left=0
final_df = final_df[final_df['balls_left']!=0]

# Split features and target
X = final_df.drop('result', axis=1)
y = final_df['result']

# Identify categorical and numeric features
categorical_cols = ['batting_team','bowling_team','city']
numeric_cols = ['runs_left','balls_left','wickets_left','total_runs','crr','rrr']

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')  # numeric columns stay as-is

# Pipeline with Logistic Regression
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit pipeline
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:,1]
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# Save the pipeline
pickle.dump(pipe, open('pipe.pkl','wb'))
