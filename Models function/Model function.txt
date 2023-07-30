def train_and_evaluate_model(X, y, model, scaler, param_grid=None, scale=None):
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # List of categorical column names to one-hot encode
    categorical_columns = ['race', 'sex', 'marital_status', 'rent_or_own',
                            'employment_status', 'census_msa',
                            'hhs_geo_region', 'employment_industry', 'employment_occupation']
    # List of column names to ordinal-encode
    ordinal_columns = ['age_group', 'education', 'income_poverty', 'opinion_seas_risk']
    # List of numerical column names
    numerical_columns = ['behavioral_antiviral_meds', 'behavioral_wash_hands',
                         'behavioral_avoidance', 'behavioral_face_mask',
                         'behavioral_large_gatherings', 'behavioral_outside_home',
                         'behavioral_touch_face', 'doctor_recc_seasonal',
                         'chronic_med_condition', 'child_under_6_months', 'health_worker',
                         'health_insurance', 'opinion_seas_vacc_effective', 'household_children',
                         'opinion_seas_sick_from_vacc', 'household_adults']
    # Define the preprocessing steps for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('numerical_imputer', SimpleImputer(strategy='median')),
    ])
    categorical_transformer = Pipeline(steps=[
        ('categorical_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    ordinal_transformer = Pipeline(steps=[
        ('ordinal_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_preprocessor', numerical_transformer, numerical_columns),
            ('categorical_preprocessor', categorical_transformer, categorical_columns),
            ('ordinal_preprocessor', ordinal_transformer, ordinal_columns)
        ])
    # Define the scaling step for Logistic Regression and KNN models
    if scale and scaler is not None:
        numerical_transformer = Pipeline(steps=[
            ('numerical_imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical_preprocessor', numerical_transformer, numerical_columns),
                ('categorical_preprocessor', categorical_transformer, categorical_columns),
                ('ordinal_preprocessor', ordinal_transformer, ordinal_columns)
            ])
    # Transform the training and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # Fit the classifier on the training data
    if param_grid is not None:
        # Create the GridSearchCV object
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', verbose=1)
        grid_search.fit(X_train_processed, y_train)
        # Use the best model from the grid search
        model = grid_search.best_estimator_
        # Print the best hyperparameters, best score, and best estimator
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
        print("Best Hyperparameters:", best_params)
        print("Best ROC-AUC Score:", best_score)
        print("Best Estimator:", best_estimator)
        # Fit the best model on the full training data
        best_estimator.fit(X_train_processed, y_train)
    # If no hyperparameter tuning, simply fit the model on the training data
    model.fit(X_train_processed, y_train)
    # Predictions on the training and test sets
    y_pred_train = model.predict(X_train_processed)
    y_pred_test = model.predict(X_test_processed)
    # Calculate the ROC AUC score
    train_roc_auc = roc_auc_score(y_train, y_pred_train)
    test_roc_auc = roc_auc_score(y_test, y_pred_test)
    # Print the ROC AUC score
    print("Train ROC AUC Score:", train_roc_auc)
    print("Test ROC AUC Score:", test_roc_auc)
    print("\n")
    # Classification report for both train and test sets
    print("Classification Report (Training Set):")
    print(classification_report(y_train, y_pred_train))
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    # Confusion matrix for both train and test sets
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Plot normalized confusion matrix for the training set
    ax1 = axes[0]
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_train_norm = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(model, X_train_processed, y_train, ax=ax1, normalize='true')
    ax1.set_title('Confusion Matrix (Training Set)')
    # Plot normalized confusion matrix for the test set
    ax2 = axes[1]
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(model, X_test_processed, y_test, ax=ax2, normalize='true')
    ax2.set_title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.show()
    # Feature importances (for RandomForest and XGBClassifier)
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        feature_importances = model.feature_importances_
        feature_names = X.columns.tolist()
        # Create a dictionary mapping feature names to importances
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        # Sort the features by importance (descending order)
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        # Print the important features
        print("Important Features:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance}")
        # Plot feature importances
        features, importances = zip(*sorted_features)
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances')
        plt.show()
    # ROC-AUC curve
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    # Append the trained model and preprocessor to the global lists
    global_trained_models.append(model)
    global_preprocessors.append(preprocessor)
    # Return the trained model and preprocessor
    return model, preprocessor