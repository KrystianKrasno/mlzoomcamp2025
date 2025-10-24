
# %%
import pickle

# %%
input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# %%
dv, model

# %%
customer = {'lead_source': 'paid_ads',
            'industry': 'education',
            'number_of_courses_viewed': 5,
            'annual_income': 52254.0,
            'employment_status': 'employed',
            'location': 'south_america',
            'interaction_count': 5,
            'lead_score': 0.49,
            'converted': 1}

# %%
X = dv.transform([customer])

# %%
model.predict_proba(X)[0,1].round(4)