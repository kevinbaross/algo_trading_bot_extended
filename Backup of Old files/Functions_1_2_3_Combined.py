#Attempt to put all functions in one .py file this is equal to 1_API_Data_Calls_Module

def data_import_cleanup_function():
    #Bitcoin historical daily prices USD
    #Source: https://data.nasdaq.com/data/BCHAIN/MKPRU-bitcoin-market-price-usd
    #btc_price = quandl.get('BCHAIN/MKPRU')
    #btc_price.rename(columns={"Value":"BTC Price"}, inplace=True)
    #btc_price.head()

    #Yfinance to replace Quandl
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(period="max")
    #btc_data = btc_data.drop
    #btc_data.head()
    
    #Etherium historical daily prices USD
    eth = yf.Ticker("ETH-USD")
    eth_data = eth.history(period="max")
    #eth_data.head()
    
    #Consumer Sentiment Data
    #Source: https://data.nasdaq.com/data/UMICH/SOC35-university-of-michigan-consumer-surveybuying-conditions-for-large-household-goods
    #consumer_sentiment = quandl.get("UMICH/SOC35")
    #consumer_sentiment.head()
    
    #Fear and Greed Index API call
    #source:https://alternative.me/crypto/fear-and-greed-index/
    response = urlopen("https://api.alternative.me/fng/?limit=0&date_format=us")
    json_data = response.read().decode('utf-8', 'replace')
    raw_fear_greed = json.loads(json_data)

    #Flatten json data
    fear_greed = pd.json_normalize(raw_fear_greed['data'])
    fear_greed = fear_greed.set_index("timestamp")
    fear_greed.index.names = ["Date"]
    fear_greed.index = pd.to_datetime(fear_greed.index)
    fear_greed = fear_greed.drop(columns="time_until_update")
    #fear_greed.tail()
    #fear_greed.head()
    
    #Combined dataframes
    combined_values = pd.concat([eth_data, fear_greed], join="inner", axis=1)
    #combined_values.head()
    
    #Testing option for merged call to streamline API calls from quandl

    #API call for merged dataset with BTC, ETH and Fear/Greed
    #merged_dataset = quandl.MergedDataset([('WIKI/AAPL', {'column_index': [11]}), ('WIKI/MSFT', {'column_index': [9,11]}), 'WIKI/TWTR'])

    #Get data for merged data set
    #data = merged_dataset.data()

    #Review data set
    #display(data)

    # Store Datafrome in memory
    #return %store combined_values


def SMAs_Function(combined_values, short_window = 20, long_window = 100, training_period_month_or_year=month, training_period=3):
    ### Step 1: Import the dataset into a Pandas DataFrame.
    # Filter the date index and close columns
    signals_df = combined_values.loc[:, ["Close"] ["value"]]

    # Use the pct_change function to generate  returns from close prices
    signals_df["Actual Returns"] = signals_df["Close"].pct_change()

    # Drop all NaN values from the DataFrame
    signals_df = signals_df.dropna()

    # Review the DataFrame
    # display(signals_df.head())
    # display(signals_df.tail())
    
    ## Step 2: Generate trading signals using short- and long-window SMA values. 
    # Set the short window and long window
    #short_window = 20
    #long_window = 100

    # Generate the fast and slow simple moving averages (4 and 100 days, respectively)
    signals_df['SMA_Fast'] = signals_df['Close'].rolling(window=short_window).mean()
    signals_df['SMA_Slow'] = signals_df['Close'].rolling(window=long_window).mean()

    signals_df = signals_df.dropna()

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())
    
    # Initialize the new Signal column
    signals_df['Signal'] = 0.0

    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())
    
    display(signals_df['Signal'].value_counts())
    
    # Calculate the strategy returns and add them to the signals_df DataFrame
    signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())
    
    # Plot Strategy Returns to examine performance
    (1 + signals_df['Strategy Returns']).cumprod().plot()
    
    ### Step 3: Split the data into training and testing datasets.
    
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna()

    # Review the DataFrame
    display(X.head())
    
    # Create the target set selecting the Signal column and assiging it to y
    y = signals_df['Signal']

    # Review the value counts
    y.value_counts()
    
    # Select the start of the training period
    training_begin = X.index.min()

    # Display the training begin date
    display(training_begin)
    
    # Select the ending period for the training data with an offset of 3 months
    training_end = X.index.min() + DateOffset(training_period_month_or_year=training_period)

    # Display the training end date
    display(training_end)
    
    # Generate the X_train and y_train DataFrames
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Review the X_train DataFrame
    X_train.head()
    
    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]

    # Review the X_test DataFrame
    X_train.head()
    
    # Scale the features DataFrames

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    return  X_train_scaled, y_train, y_test


# inputs to the function
# classifier_name =  svm.SVC()
# classifier_name =  LogisticRegression()
# need to X_train_scaled, y_train


def classifier_function(classifier_name, X_train_scaled, y_train, y_test):
    # Initiate the model instance
    model_instance = classifier_name
    
    # Fit the model to the data using the training data
    model = model_instance.fit(X_train_scaled, y_train)
    
    # Use the testing data to make the model predictions
    pred = model.predict(X_test_scaled)

    # Review the model's predicted values
    #pred[:10]
    
    # Review the classification report associated with the `classifier_name` model predictions. 
    # Use a classification report to evaluate the model using the predictions and testing data
    model_testing_report =  classification_report(y_test, pred)
       
    # Print the classification report
    #print(svm_testing_report)
    
    # Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.
    # Create a new empty predictions DataFrame.
        # Create a predictions DataFrame
    new_predictions_df = pd.DataFrame(index=X_test.index)

    # Add the classifier_name model predictions to the DataFrame
    new_predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    new_predictions_df['Actual Returns'] = signals_df['Actual Returns'] 
    
    # Add the strategy returns to the DataFrame
    new_predictions_df['Strategy Returns'] = (new_predictions_df['Actual Returns'] * new_predictions_df['Predicted'])

    # Review the DataFrame
    # display(new_predictions_df.head())
    # display(new_predictions_df.tail())

    # Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.
    # Plot the actual returns versus the strategy returns
    cumulative_return_plot = (1 + new_predictions_df[["Actual Returns", "Strategy Returns"]]).cumprod().plot()
    
    return model_testing_report, cumulative_return_plot