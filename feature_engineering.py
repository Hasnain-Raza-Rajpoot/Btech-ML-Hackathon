import pandas as pd

train = pd.read_csv('floodsense_cleaned.csv')
elev  = pd.read_csv('D:\\BNU\\upload\\district_elevation_reference.csv')


train = train.merge(elev[['district', 'avg_elevation_m']], on='district', how='left')


district_map = {
    'Sindh_District': 0,
    'Balochistan_District': 1,
    'KP_District': 2
}
train['district_encoded'] = train['district'].map(district_map)


train['date'] = pd.to_datetime(train['date'], dayfirst=False)
train = train.sort_values(['district', 'date']).reset_index(drop=True)


for district in train['district'].unique():
    mask = train['district'] == district
    train.loc[mask, 'precip_lag1'] = train.loc[mask, 'precipitation'].shift(1)
    train.loc[mask, 'precip_lag2'] = train.loc[mask, 'precipitation'].shift(2)
    train.loc[mask, 'precip_lag3'] = train.loc[mask, 'precipitation'].shift(3)
    train.loc[mask, 'soil_lag1']   = train.loc[mask, 'soil_moisture'].shift(1)
    train.loc[mask, 'water_lag1']  = train.loc[mask, 'water_area_km2'].shift(1)

lag_cols = ['precip_lag1','precip_lag2','precip_lag3','soil_lag1','water_lag1']
train[lag_cols] = train[lag_cols].fillna(0)


train['rain_x_soil']        = train['precipitation'] * train['soil_moisture']
train['rain_x_elevation']   = train['precipitation'] * train['avg_elevation_m']
train['soil_x_water']       = train['soil_moisture'] * train['water_area_km2']
train['precip_7day_x_soil'] = train['precip_7day_avg'] * train['soil_moisture']


train.to_csv('floodsense_engineered.csv', index=False)
print(f"Done! Shape: {train.shape}")
print(f"Any NaNs: {train.isnull().sum().sum()}")