# DriverPass System UML Diagrams

## Use Case Diagram

# DriverPass System UML Diagrams

## Use Case Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'useCaseBorderColor': '#6e6e6e', 'useCaseBackgroundColor': '#f9f9f9', 'actorBorderColor': '#000000', 'actorTextColor': '#000000', 'actorBackgroundColor': '#ffffff', 'fontFamily': 'Arial', 'primaryColor': '#ffcc00', 'primaryTextColor': '#000000', 'secondaryColor': '#e8e8e8', 'tertiaryColor': '#b2b2b2', 'lineColor': '#333333', 'textColor': '#333333' }}}%%
flowchart LR
    subgraph Actors
        Student
        Instructor
        Admin
        Secretary
    end
    subgraph Use Cases
        Register_for_Course
        Take_Practice_Test
        Schedule_Driving_Test
        Modify_Reservation
        View_Schedule
        Add_Comments
        Manage_Accounts
        Generate_Reports
        Schedule_Appointment
        Cancel_Appointment
    end

    Student --> Register_for_Course
    Student --> Take_Practice_Test
    Student --> Schedule_Driving_Test
```

## Activity Diagram for "Register for Course"

```mermaid
%%{init: {'theme': 'forest'}}%%
flowchart TD
    A[Start] --> B[Fill Registration Form]
    B --> C[Submit Form]
    C --> D[Receive Confirmation]
    D --> E[End]
```
## Activity Diagram for "Take Practice Test"
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'startColor': '#ffcc00', 'startTextColor': '#000000', 'endColor': '#ffcc00', 'endTextColor': '#000000', 'fontFamily': 'Arial', 'primaryColor': '#e8e8e8', 'primaryTextColor': '#000000', 'lineColor': '#333333', 'textColor': '#333333' }}}%%
flowchart TD
    A[Start] --> B[Log into Account]
    B --> C[Select Practice Test]
    C --> D[Start Test]
    D --> E[Answer Questions]
    E --> F[Submit Test]
    F --> G[Receive Results]
    G --> H[End]
```

## Sequence Diagram for "Register for Course"

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'actorBorderColor': '#000000', 'actorBackgroundColor': '#ffffff', 'fontFamily': 'Arial', 'primaryColor': '#ffcc00', 'primaryTextColor': '#000000', 'secondaryColor': '#e8e8e8', 'tertiaryColor': '#b2b2b2', 'lineColor': '#333333', 'textColor': '#333333' }}}%%
sequenceDiagram
    participant Student
    participant System
    participant QuestionDB as "Question Database"

    Student ->> System: Log into Account
    Student ->> System: Select Practice Test
    System ->> QuestionDB: Fetch Test Questions
    QuestionDB -->> System: Return Test Questions
    System -->> Student: Display Test Questions
    Student ->> System: Submit Answers
    System ->> QuestionDB: Store Answers
    QuestionDB -->> System: Confirm Storage
    System -->> Student: Display Results
```

## Class Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'classBorderColor': '#000000', 'classTextColor': '#000000', 'classBackgroundColor': '#ffffff', 'fontFamily': 'Arial', 'primaryColor': '#e8e8e8', 'primaryTextColor': '#000000', 'lineColor': '#333333', 'textColor': '#333333' }}}%%
classDiagram
    class User {
        +int userID
        +string name
        +string email
        +string password
        +string address
    }
    
    class Student {
        +int studentID
        +string licenseNumber
        +Date dateOfBirth
    }
    
    class Instructor {
        +int instructorID
        +string specialization
        +int yearsOfExperience
    }
    
    class Admin {
        +int adminID
        +string role
    }
    
    class Course {
        +int courseID
        +string title
        +string description
        +Date startDate
        +Date endDate
    }
    
    class Test {
        +int testID
        +string testName
        +string status
        +int score
        +Date dateTaken
    }
    
    class Appointment {
        +int appointmentID
        +Date appointmentDate
        +Time appointmentTime
    }
    
    User <|-- Student
    User <|-- Instructor
    User <|-- Admin
    Student --> Course : registers for
    Student --> Test : takes
    Instructor --> Course : teaches
    Admin --> Course : manages
    Student --> Appointment : schedules
    Instructor --> Appointment : attends
```
