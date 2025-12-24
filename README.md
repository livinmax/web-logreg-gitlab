<img width="1395" height="794" alt="image" src="https://github.com/user-attachments/assets/80dd4b41-bc28-4a0b-97e2-4ed225590693" />


Business Case
- The objective of this project was to develop a web-based decision support tool to improve sales efficiency. The application captures specific customer event data to predict the likelihood of a customer accepting or declining a commercial proposal, allowing teams to prioritize high-value leads.

Technical Solution
- User Interface: A web-based entry point where customer service officers input event-specific data via a structured form.
- Data Pipeline: The application automates data collection and preprocessing (transformation), ensuring features are formatted for model consumption.
- Machine Learning: The system triggers a Logistic Regression model to perform binary classification on the input data.
- Real-time Feedback: The generated prediction is instantly transmitted back to the web interface to assist the officer in real-time decision-making.

Deployment & Infrastructure
- CI/CD Pipeline: Utilized GitLab for version control and the automation of model training and deployment cycles.
- Containerization: The application and its dependencies were containerized to ensure environment consistency.
- Cloud Hosting: Deployed via a Cloud Virtual Machine, utilizing a Container Registry for secure storage and management of the application and model images.

Project structure
- .gitlab-ci.yml
- service_model/
-   model_train.py
-   requirements.txt
-   Dockerfile
- service_web/
-   main.py
-   requirements.txt
-   Dockerfile
-   templates/
-     index.html
