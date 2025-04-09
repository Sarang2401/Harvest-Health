#include <Servo.h>
#define SERVO_PIN 9
Servo myservo;  // create servo object to control a servo
#define TRIGGER_PIN 13  // NodeMCU Pin D1 connected to (Trigger) pin of HC-SR04
#define ECHO_PIN 15     // NodeMCU Pin D2 connected to (Echo) pin of HC-SR04
#define sensorPin 12    // Pin connected to the LM393 output

// Define the pins for the motor
#define MOTOR_PIN1 5
#define MOTOR_PIN2 4
#define MOTOR_PIN3 0
#define MOTOR_PIN4 2
#define MOTOR_ENA 16
#define MOTOR_ENB 14

bool motorStopped = false;

float measured_distance; // Distance covered per pulse (change according to your sensor)
volatile unsigned int pulseCount = 0;
bool vehicleDetected = false;  // Flag to track vehicle detection

void ICACHE_RAM_ATTR pulseCounter() {
  pulseCount++;
  vehicleDetected = true;  // Set the flag when a vehicle is detected
}

void setup() {
  Serial.begin(9600);
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(sensorPin, INPUT);
  myservo.attach(SERVO_PIN);  // attaches the servo on pin 9 
  attachInterrupt(digitalPinToInterrupt(sensorPin), pulseCounter, RISING);
  

  // Initialize the motor pins as output
  pinMode(MOTOR_PIN1, OUTPUT);
  pinMode(MOTOR_PIN2, OUTPUT);
  pinMode(MOTOR_PIN3, OUTPUT);
  pinMode(MOTOR_PIN4, OUTPUT);
  pinMode(MOTOR_ENA, OUTPUT);
  pinMode(MOTOR_ENB, OUTPUT);
}

void loop() {
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH);
  float distance = (duration / 2) * 0.0344;
  
  Serial.print("Distance (Ultrasonic): ");
  Serial.print(distance);
  Serial.println(" cm");

  // If the distance is less than 10 cm, stop the motor
  if (distance < 10) {
    digitalWrite(MOTOR_PIN1, LOW);
    digitalWrite(MOTOR_PIN2, LOW);
    digitalWrite(MOTOR_PIN3, LOW);
    digitalWrite(MOTOR_PIN4, LOW);
  } 
  else if(measured_distance >= 50) {
    // Otherwise, run the motor
    digitalWrite(MOTOR_PIN1, LOW);
    digitalWrite(MOTOR_PIN2, LOW);
    digitalWrite(MOTOR_PIN3, LOW);
    digitalWrite(MOTOR_PIN4, LOW);
  }
  else if (!motorStopped){
      digitalWrite(MOTOR_PIN1, HIGH);
    digitalWrite(MOTOR_PIN2, LOW);
    digitalWrite(MOTOR_ENA, HIGH);
    digitalWrite(MOTOR_ENB, HIGH);
    digitalWrite(MOTOR_PIN3, HIGH);
    digitalWrite(MOTOR_PIN4, LOW);
  }
   

  if (vehicleDetected) {
    measured_distance = ((pulseCount / 10) * 3.142 * 0.7);  // Convert time to seconds
    Serial.print("Distance (LM393): ");
    Serial.print(measured_distance);
    Serial.println(" cm");
    vehicleDetected = false; // Reset the vehicle detection flag
    
    myservo.write(90); // rotate servo 90 degrees
    delay(10000); // wait for 10 seconds
    myservo.write(0); // retract servo
    
  } else {
    Serial.println("No vehicle detected");
  }
  pulseCount++;
  attachInterrupt(digitalPinToInterrupt(sensorPin), pulseCounter, RISING);
  delay(1000);
}