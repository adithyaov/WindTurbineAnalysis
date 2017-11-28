# After installing the dependencies

execfile('../../src/windSpeedVsActivePower.py')
WSvsAPMain = main
execfile('../../src/rotorSpeedVsActivePower.py')
RSvsAPMain = main

print WSvsAPMain('/home/xxx/yyy/zzz.csv', plotGraph=True)
