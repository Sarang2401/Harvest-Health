#include <Wire.h> 
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Pin Definitions
#define DHTPIN 13       // Digital pin connected to the DHT sensor
#define DHTTYPE DHT11   // Define sensor type as DHT11
#define SOIL_MOISTURE_PIN 12 // Pin connected to the soil moisture sensor
#define LDR_PIN 15 // Pin connected to the Light Dependent Resistor (LDR)

DHT_Unified dht(DHTPIN, DHTTYPE); // Initialize the DHT sensor
uint32_t delayMS; // Variable to store the sensor's delay time

// WiFi and ThingSpeak Configuration
String apiKey = " "; // Add your ThingSpeak API Key
const char* ssid = ""; // Add your WiFi SSID
const char* password = ""; // Add your WiFi password

void setup() {
  Serial.begin(115200); // Start the serial monitor
  delay(10);

  // Initialize DHT sensor
  dht.begin();
  sensor_t sensor;
  dht.temperature().getSensor(&sensor); // Get temperature sensor details
  dht.humidity().getSensor(&sensor); // Get humidity sensor details
  delayMS = sensor.min_delay / 1000; // Minimum delay between sensor readings

  // Connect to WiFi network
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { // Retry until connected
    delay(500);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  delay(delayMS); // Wait for the required delay
  sensors_event_t event;

  // Read temperature and humidity from DHT sensor
  dht.temperature().getEvent(&event); 
  float temperature = event.temperature;
  dht.humidity().getEvent(&event);
  float humidity = event.relative_humidity;

  // Read soil moisture level from analog pin
  int soilMoistureValue = analogRead(SOIL_MOISTURE_PIN);
  Serial.println(soilMoistureValue); // Print raw value to Serial Monitor
  delay(1000);

  // Convert raw soil moisture value to percentage
  float soilMoisturePercentage = map(soilMoistureValue, 0, 4095, 0, 100);

  // Read light intensity value from LDR sensor
  int ldrValue = digitalRead(LDR_PIN); // Assumes LDR is configured as digital input
  Serial.println(ldrValue); // Print raw LDR value to Serial Monitor
  delay(1000);

  // Convert LDR raw value to percentage representation
  float lightIntensityPercentage = map(ldrValue, 0, 1, 0, 100);

  // Check if WiFi connection is active
  if(WiFi.status()== WL_CONNECTED){
    HTTPClient http;
    // Prepare HTTP GET request for ThingSpeak
    http.begin("http://api.thingspeak.com/update?api_key="+apiKey+"&field1="+String(temperature)+"&field2="+String(humidity)+"&field3="+String(soilMoisturePercentage)+"&field4="+String(ldrValue));
    int httpResponseCode = http.GET(); // Send GET request

    if(httpResponseCode>0){ // Success
      String response = http.getString(); // Get server response
      Serial.println(httpResponseCode); // Print HTTP response code
      Serial.println(response); // Print response
    }else{ // Failure
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
    }
    http.end(); // End HTTP request
  }else{
    Serial.println("Error in WiFi connection"); // WiFi not connected
  }
  delay(10000); // Delay before the next set of readings and HTTP request
}
