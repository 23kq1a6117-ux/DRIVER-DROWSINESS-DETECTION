// Receives '1' (drowsy -> alert ON) or '0' (alert OFF) over serial at 9600 baud.
// Connect an LED, buzzer, or relay driver to ALERT_PIN (prototype output only).
const int ALERT_PIN = 8;

void setup() {
  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') digitalWrite(ALERT_PIN, HIGH);
    else if (c == '0') digitalWrite(ALERT_PIN, LOW);
  }
}
