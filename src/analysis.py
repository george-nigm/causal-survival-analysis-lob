import pandas as pd 

# with open('conf/create_clean_data.yaml', 'r') as file:
#     conf = yaml.safe_load(file)

# with open('conf/create_clean_data.yaml', 'r') as file:
#     prices = yaml.safe_load(file)

if __name__ == "__main__":

    
    prices = pd.read_pickle('cleaned/pinnacle.pkl')

    print()
    print(prices.head())
    print()