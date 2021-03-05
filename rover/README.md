# Rover

## Components

### Chassis

The base of the rover is the "TS100 Shock Absorber Tank Chassis" by [DOIT](www.doit.am) ([manual](https://raw.githubusercontent.com/SeeedDocument/Outsourcing/master/110090267%20TS100%20shock%20absorber%20tank%20chassis%20with%20track%20and%20DC%20geared%20motors%20Kit/InstallationforTS100%20.pdf)). This kit for this rover was rover was kindly provided to us by our mentor at [Airobot](https://airobot.eu/).

The chassis uses [simple DC motors](https://item.taobao.com/item.htm?spm=a1z10.5-c.w4002-7420481794.72.fWWJc1&id=45203541487) with a Hall-effect sensor. It is possible to use the sensor with Ardupilot, but we didn't in our setup.

### H-bridge

To control the speed and direction of the two motors we used an [H-bridge breakout board](https://www.velleman.eu/products/view/?id=435576), also provided to us by Airobot.

To ease control of the H-bridge through software we added a simple [NOR-gate](https://web.mit.edu/6.131/www/document/7402.pdf) circuit.

TODO: Finish diagram
