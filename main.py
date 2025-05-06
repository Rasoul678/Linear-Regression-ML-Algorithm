# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


def main():
    print("Linear Regression Implementation")

    # Generate a simple regression dataset
    X, y = datasets.make_regression(n_samples=100, n_features=1 , noise=20, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X_train, y_train)

    reg = LR()
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)
    y_pred2 = reg.predict(X_test)


    # Calculate error
    mse = LinearRegression.mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    mse2 = LinearRegression.mean_squared_error(y_test, y_pred2)
    print(f"Mean Squared Error: {mse2}")

    # Plot results
    plt.figure(figsize=(7,5))
    cmap = plt.get_cmap()
    plt.scatter(X_train, y_train, color=cmap(0.9), label='Actual Train', s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), label='Actual Test', s=10)
    plt.plot(X_test, y_pred2, color='black', linewidth=1, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.savefig("linear.png")

if __name__ == "__main__":
    main()
