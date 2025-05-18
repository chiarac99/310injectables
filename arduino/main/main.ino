/*
main.ino

Main Arduino file for 310 Injectables

This file should handle all comms with electronics. These include:
- limit switches for linear actuator positioning
- Nema 17 motors for syringe alignment subsystem and two linear actuators

*/

#define DIR_PIN_ALIGN 12 // linear actuator that moves aligning steps up and down
#define DIR_PIN_CUT 9 // linear actuator that moves cutting step up and down
#define STEP_PIN_ALIGN 11 
#define STEP_PIN_CUT 8

#define LIMIT_SWITCH_ALIGN 4 // limit switch for positioning linear actuator 1
#define LIMIT_SWITCH_CUT 2 // limit switch for positioning linear actuator 2

/*
LINEAR ACTUATOR POSITIONING
- linear actuator position is in integer number of steps from the zero position
- the zero position is the position of the limit switch at the top its travel
- it should hit this point once every cycle (when going up)
*/
#define microStepsPerRotation 200 // 5mm pitch of lead screw
const int StepsPerAlignCycle = int(microStepsPerRotation*5.08); // number of steps to do at least 1" of travel (from zero position) - 5.08 rotations
const int StepsPerCutCycle = int(microStepsPerRotation*10.16); // number of steps to do at least 2" of travel - 10.16 rotations

#define actuatorUpDir true
#define actuatorDownDir false
bool direction = actuatorUpDir; // start direction upwards
uint8_t alignActivated = 1; // is the actuator active for this loop
uint8_t cutActivated = 1;
#define ActuatorStepDelay 500  // delay btw HIGH and LOW signal for align and cutting stepper motors

int alignStepCount = 0; // step count record
int cutStepCount = 0;

// TODO: temp stuff
bool lastSwitchState = HIGH;
const int switchPin = 2;  // Limit switch


// states
typedef enum {
  INITIALISING_CAMERA,
  INITIALISING_BLADE,
  MOVING_UP, // no need to home actuators because we will always be moving upwards first; we check if we reach the limit switch at the top of EVERY cycle
  MOVING_DOWN,
  ASK_FOR_IMG,
  WAITING_FOR_IMG_OUTPUT,
} systemState_t;
systemState_t systemState = MOVING_UP;

void setup() {
  Serial.begin(115200);
  while (Serial.available()) Serial.read();  // Clear buffer

  // linear actuators
  pinMode(DIR_PIN_ALIGN, OUTPUT);
  pinMode(DIR_PIN_CUT, OUTPUT);
  pinMode(STEP_PIN_ALIGN, OUTPUT);
  pinMode(STEP_PIN_CUT, OUTPUT);

  digitalWrite(DIR_PIN_ALIGN, direction); // initialise dir of align actuator
  digitalWrite(DIR_PIN_CUT, direction); // initialise dir of cut actuator

  // limitswitches
  pinMode(LIMIT_SWITCH_ALIGN, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_CUT, INPUT_PULLUP);

  delay(2000);

}

void loop() {

  // update state
  stateHandler();

  // move step/cut actuators if appropriate
  moveActuators();

  // if (checkLimitSwitch(LIMIT_SWITCH_1)){
  //   Serial.println("limit switch pressed!");
  // }

}

void stateHandler(){
  // state machine
  switch (systemState) {

      case INITIALISING_CAMERA:
        // let python know that we're ready to start the camera
        // may need two states bc then we need to wait for python to tell us okay
        break;

      case INITIALISING_BLADE:
        // turn on motor for rotary saw
        break;

      case MOVING_UP:
        // move actuators up until both activate top limit switches
        if (movedUp()) systemState = ASK_FOR_IMG;
        break;

      case MOVING_DOWN:
        // move actuators down until both steppers do the correct number of StepsPerCycle
        if (movedDown()) {
          Serial.println("moving back up!");
          // reactivated actuators
          alignActivated = 1;
          cutActivated = 1;

          // restart step counts
          alignStepCount = 0;
          cutStepCount = 0;

          // change state
          systemState = MOVING_UP;
        }
        break;

      case ASK_FOR_IMG:
        Serial.println("SNAP"); // send msg to python to take image
        systemState = WAITING_FOR_IMG_OUTPUT;
        break;

      case WAITING_FOR_IMG_OUTPUT:
        if (receivedImgOutput()) {
          Serial.println("received output!");

          // activate steppers
          alignActivated = 1;
          cutActivated = 1;

          // change state
          systemState = MOVING_DOWN;
        }
        break;

      default:

        // statements
        Serial.println("this is not a state :'(");
        break;

    }


}


/* state functions*/
uint8_t movedUp(){
  // return true if both limitswitches have been activated

  // bools for recording whether alignment and cutting steps have reached the end of either their upward or downward cycle
  uint8_t aligned = limitswitchActivated(LIMIT_SWITCH_ALIGN);
  uint8_t cut = limitswitchActivated(LIMIT_SWITCH_CUT);

  // set dir
  direction = actuatorUpDir;

  // deactivate the stepper if limitswitches were activated
  if (aligned) alignActivated = 0; 
  if (cut) cutActivated = 0;

  // return true if both actuators have reached their endpoints
  if (!alignActivated && !cutActivated) return 1;

  // otherwise keep moving align actuator up
  return 0;
}

uint8_t movedDown(){
  // return true if completed the correct number of steps for a cycle
  
  // set dir
  direction = actuatorDownDir;

  // get out if both actuators are deactivated
  if (!alignActivated && !cutActivated) return 1;

  // deactivate actuators if step count reached
  if (alignStepCount >= StepsPerAlignCycle) alignActivated = 0;
  if (cutStepCount >= StepsPerCutCycle) cutActivated = 0;

  // increment step count
  alignStepCount++;
  cutStepCount+=2; // cutting step is moving twice as fast as align step

  return 0;

}

void moveActuators(void){
  // update direction
  digitalWrite(DIR_PIN_ALIGN, direction);
  digitalWrite(DIR_PIN_CUT, direction);

  // update step
  if (alignActivated || cutActivated) {
    // Step 1: Leading edge
    if (alignActivated) digitalWrite(STEP_PIN_ALIGN, HIGH);
    if (cutActivated)   digitalWrite(STEP_PIN_CUT, HIGH);
    delayMicroseconds(ActuatorStepDelay);

    // Step 2: Trailing edge
    digitalWrite(STEP_PIN_ALIGN, LOW);
    digitalWrite(STEP_PIN_CUT, LOW);
    delayMicroseconds(ActuatorStepDelay);

    // Step 3: Extra step for cut actuator to make it 2x faster
    if (cutActivated) {
      digitalWrite(STEP_PIN_CUT, HIGH);
      delayMicroseconds(ActuatorStepDelay);
      digitalWrite(STEP_PIN_CUT, LOW);
      delayMicroseconds(ActuatorStepDelay);
    }
  }

  // // update step
  // // depends on globals alignActivated, cutActivated
  // if(alignActivated) digitalWrite(STEP_PIN_ALIGN, HIGH);
  // if(cutActivated) digitalWrite(STEP_PIN_CUT, HIGH);

  // if(alignActivated || cutActivated){
  //   delayMicroseconds(ActuatorStepDelay);
  //   digitalWrite(STEP_PIN_ALIGN, LOW);
  //   digitalWrite(STEP_PIN_CUT, LOW);
  //   delayMicroseconds(ActuatorStepDelay);  // Delay between steps (increase for slower speed)
  // }

  // if neither are activated, do nothing

}


uint8_t limitswitchActivated(uint8_t pin){
  // return true if limit switch at specified pin is activated
  if (!digitalRead(pin)){
    return 1;
  }
  return 0;
}


uint8_t receivedImgOutput(void) {
  static String buf = "";

  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      if (buf.indexOf("OUTPUT") >= 0) {
        buf = "";
        return 1;
      }
      buf = ""; // Reset even if message was invalid
    } else {
      buf += inChar;
    }
  }
  return 0;
}
