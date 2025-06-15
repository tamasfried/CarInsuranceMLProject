# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def prompt_user_for_features():
    # Function to collect driver and vehicle information from user input
    print("\nEnter driver and car information:")
    driver_age = float(input("Driver Age: "))
    driver_experience = float(input("Driver Experience (years): "))
    previous_accidents = float(input("Previous Accidents: "))

    # Convert annual mileage from kilometres to thousands of kilometres to match dataset format
    annual_mileage = float(input("Annual Mileage (km): ")) / 1000
    car_manufacturing_year = float(input("Car Manufacturing Year: "))

    # Calculate car age automatically using current year
    current_year = pd.Timestamp.now().year
    car_age = current_year - car_manufacturing_year

    # Return features as a nested list for DataFrame creation
    return [[driver_age, driver_experience, previous_accidents, annual_mileage, car_age]]

if __name__ == "__main__":
    while True:

        # Display main menu options
        print("\nMain Menu:")
        print("1. Analyse dataset and visualise results")
        print("2. Predict insurance premium for a new driver/car")
        print("3. Exit")
        mode = input("Enter 1, 2, or 3: ").strip()

        # Exit condition
        if mode == '3':
            print("Exiting program.")
            break

        # Load and prepare the dataset
        df = pd.read_csv('car_insurance.csv')

        # Separate features (X) and target variable (y)
        X = df.drop(['Insurance Premium ($)', 'Car Manufacturing Year'], axis=1) # Drop Insurance Premium and Car Manufacturing Year for training as car age is already defined.
        y = df['Insurance Premium ($)']

        # Split data into training and testing sets with 80-20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialise and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        if mode == '1':
            # Generate predictions for test set
            y_pred = model.predict(X_test)

            # Calculate model performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display model performance metrics
            print("\nModel Performance:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R-squared Score: {r2:.2f}")
            
            # Display feature coefficients to show their impact on predictions
            print("\nFeature Coefficients:")
            for feature, coef in zip(X.columns, model.coef_):
                print(f"{feature}: {coef:.4f}")
            
            # Create and save correlation matrix visualisation
            plt.figure(figsize=(10, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Features')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            
            # Create and save actual vs predicted values plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Premium')
            plt.ylabel('Predicted Premium')
            plt.title('Actual vs Predicted Insurance Premiums')
            plt.tight_layout()
            plt.savefig('actual_vs_predicted.png')
            
            # Create and save residual plot to check model assumptions
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Premium')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.tight_layout()
            plt.savefig('residual_plot.png')
            
        elif mode == '2':
            # Collect user input for prediction
            user_features = prompt_user_for_features()

            # Create df with proper feature names matching training data
            user_df = pd.DataFrame(user_features, columns=['Driver Age', 'Driver Experience', 'Previous Accidents', 
                                                         'Annual Mileage (x1000 km)', 'Car Age'])
            
            # Generate prediction using trained model
            predicted_premium = model.predict(user_df)[0]
            
            # Display detailed analysis of the prediction
            print("\nPrediction Analysis:")
            print("-------------------")

            # Show base premium (intercept)
            base_premium = model.intercept_
            print(f"Base Premium: ${base_premium:.2f}")
            
            # Show how each feature contributes to the final premium
            print("\nFeature Contributions:")
            for feature, coef, value in zip(X.columns, model.coef_, user_features[0]):
                contribution = coef * value
                print(f"{feature}: ${contribution:.2f} ({'+' if contribution > 0 else ''}{contribution:.2f})")
            
            # Display final predicted premium
            print("\nFinal Prediction:")
            print(f"Predicted Insurance Premium: ${predicted_premium:.2f}")
            
        else:
            # Handle invalid menu selection
            print("Invalid selection. Please choose 1, 2, or 3.") 