#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Servo.h>

const char* ssid = "AARAV";
const char* password = "12345678";

ESP8266WebServer server(80);
Servo myservo;

int soilMoisturePin = A0; // Analog pin connected to soil moisture sensor
int servoPin = D1;        // Digital pin connected to servo motor
int thresholdValue = 530; // Threshold value for soil moisture

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  myservo.attach(servoPin);
  server.on("/", handleRoot);
  server.on("/moisture", handleMoisture);
  server.begin();
}

void loop() {
  server.handleClient();
}

void handleRoot() {
  String page = "<html><body><h1>Servo Control</h1><p>Click to control servo:</p><a href=\"/moisture\">Read Moisture and Actuate Servo</a></body></html>";
  server.send(200, "text/html", page);
}

void handleMoisture() {
  int moistureValue = analogRead(soilMoisturePin);
  Serial.print("Soil Moisture: ");
  Serial.println(moistureValue);

  if (moistureValue > thresholdValue) {
    myservo.write(90); // Rotate servo to dig into the soil
    Serial.println("Dry soil detected - Servo actuated");
  } else {
    myservo.write(0); // Rotate servo back to initial position
    Serial.println("Wet soil detected - Servo returned to initial position");
  }
  server.send(200, "text/plain", "Soil moisture read and servo actuated");
}
