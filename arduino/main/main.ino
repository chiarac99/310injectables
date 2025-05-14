/*
main.ino

Main Arduino file for 310 Injectables

This file should handle all comms with electronics. These include:
- limit switches for linear actuator positioning
- Nema 17 motors for syringe alignment subsystem and two linear actuators

*/

#define LINEAR_ACT_ALIGN  // linear actuator that moves aligning steps up and down
#define LINEAR_ACT_CUT  // linear actuator that moves cutting step up and down
#define LIMIT_SWITCH_ALIGN  2 // limit switch for positioning linear actuator 1
#define LIMIT_SWITCH_CUT  // limit switch for positioning linear actuator 2

/*
LINEAR ACTUATOR POSITIONING
- linear actuator position is in integer number of steps from the zero position
- the zero position is the position of the limit switch at the top its travel
- it should hit this point once every cycle (when going up)
*/
const int StepsPerAlignCycle = 3; // number of steps to do 1" of travel (from zero position)
const int StepsPerCutCycle = 5; // number of steps to do 2" of travel




// states
typedef enum {
  INITIALISING_CAMERA,
  INITIALISING_BLADE,
  MOVING_UP, // no need to home actuators because we will always be moving upwards first; we check if we reach the limit switch at the top of EVERY cycle
  MOVING_DOWN,
  ASK_FOR_IMG,
  WAITING_FOR_IMG_OUTPUT,
} systemState_t;
systemState_t systemState = HOMING_ACTUATORS;

void setup() {
  Serial.begin(115200);
  Serial.println("Setup starting...");

  // limitswitch setup
  pinMode(LIMIT_SWITCH_1, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_1, INPUT_PULLUP);

}

void loop() {
  
  runState();

  if (checkLimitSwitch(LIMIT_SWITCH_1)){
    Serial.println("limit switch pressed!");
  }

}

void runState(){
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
        if (movedUp()) systemState = MOVING_DOWN; // TODO change this to take image?
        break;

      case MOVING_DOWN:
        // move actuators down until both steppers do the correct number of StepsPerCycle
        if (movedDown()) systemState = MOVING_UP;
        break;

      case ASK_FOR_IMG:
        break;

      case WAITING_FOR_IMG_OUTPUT:
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
  uint8_t aligned = checkLimitSwitch(LIMIT_SWITCH_ALIGN);
  unint8_t cut = checkLimitSwitch(LIMIT_SWITCH_CUT);

  if (aligned && cut) return 1;

  if (!aligned){
    // keep moving align actuator up
    
  }

  if(!cut){
    // keep moving cut actuator up
  }

  

}


uint8_t checkLimitSwitch(uint8_t pin){
  // return true if limit switch at specified pin is activated
  if (!digitalRead(pin)){
    return 1;
  }
  return 0;
}
