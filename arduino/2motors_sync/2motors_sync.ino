const int stepPin1 = 8;
const int dirPin1 = 9;
const int stepPin2 = 11;
const int dirPin2 = 12;

const unsigned long TOTAL_DURATION = 2000000UL; // 2 seconds
const int steps1 = 1600;
const int steps2 = 2200;

unsigned long interval1 = TOTAL_DURATION / steps1;
unsigned long interval2 = TOTAL_DURATION / steps2;

unsigned long lastStepTime1 = 0;
unsigned long lastStepTime2 = 0;

unsigned long pulseStart1 = 0;
unsigned long pulseStart2 = 0;

bool pulseHigh1 = false;
bool pulseHigh2 = false;

int currentStep1 = 0;
int currentStep2 = 0;

bool direction = false;

void setup() {
  pinMode(stepPin1, OUTPUT);
  pinMode(dirPin1, OUTPUT);
  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin2, OUTPUT);

  digitalWrite(dirPin1, direction);
  digitalWrite(dirPin2, direction);
}

void loop() {
  unsigned long now = micros();

  // STEP control for motor 1
  if (!pulseHigh1 && currentStep1 < steps1 && now - lastStepTime1 >= interval1) {
    digitalWrite(stepPin1, HIGH);
    pulseStart1 = now;
    pulseHigh1 = true;
    lastStepTime1 = now;
    currentStep1++;
  } else if (pulseHigh1 && now - pulseStart1 >= 10) {
    digitalWrite(stepPin1, LOW);
    pulseHigh1 = false;
  }

  // STEP control for motor 2
  if (!pulseHigh2 && currentStep2 < steps2 && now - lastStepTime2 >= interval2) {
    digitalWrite(stepPin2, HIGH);
    pulseStart2 = now;
    pulseHigh2 = true;
    lastStepTime2 = now;
    currentStep2++;
  } else if (pulseHigh2 && now - pulseStart2 >= 10) {
    digitalWrite(stepPin2, LOW);
    pulseHigh2 = false;
  }

  // Check for completion
  if (currentStep1 >= steps1 && currentStep2 >= steps2 && !pulseHigh1 && !pulseHigh2) {
    // Flip direction
    direction = !direction;
    digitalWrite(dirPin1, direction);
    digitalWrite(dirPin2, direction);

    // Reset step counters and timers
    currentStep1 = 0;
    currentStep2 = 0;
    lastStepTime1 = now;
    lastStepTime2 = now;

    delay(250);  // Optional pause between direction change
  }
}