# Autonomously mapping linear landscape features using a drone

nl: **Autonoom in kaart brengen van lineaire landschapselement met behulp van een drone**

# 1. Requirements


## 1.1. Background

Mapping linear landscape features (paths, streams...) can be challenging, especially in forests. Orthophotos are often blocked by tree cover and GPS reception is usually poor. Accurately mapping those features at scale is thus at best labour intensive, and at worst unfeasible.


## 1.2. Opportunity

We will be investigating the possibility of using a semi-remote-controlled drone/UAV to automatically map linear landscape features &quot;on the ground&quot; (below the forest canopy, above the landscape features). We believe this approach has the possibility of making the mapping of a large forest as easy as designating the area to be mapped and launching the drone.


## 1.3. Outline

Our investigation will consist of five stages, each stage dependent on the viability of the last. The first stage will be using a neural network to segment images into &#39;flyable&#39; and &#39;non-flyable&#39; regions. The second stage will consist of extracting steering data from the segmented images. The third and fourth stage will be autonomously steering a rover and drone respectively over a forest path. The fifth and final stage will be the introduction of SLAM (simultaneous localization and mapping) into the system. In this stage we will use, among other components, a flow sensor and the drone&#39;s IMU to build a map and to make routing decisions in real-time.


## 1.4. Milestones

1. Semantic image segmentation  
  1.1. Data collected  
  1.2. Data labelled  
  1.3. Data pipeline  
  1.4. Basic segmentation CNN  
  1.5. Real-world CNN validation  
  1.6. Segmentation CNN comparison
2. Steering from segmented image  
  2.1. Single image  
  2.2. ROS integration  
  2.3. Real-world test (walking + SBC)
3. Autonomous rover  
  3.1. Hardware built  
  3.2. Manual control  
  3.3. Control with ROS  
  3.4. Steering integration in ROS
4. Autonomous drone  
  4.1. Hardware built  
  4.2. Manual control  
  4.3. Control with ROS  
  4.4. Steering integration in ROS
5. SLAM  
  5.1. Top-down map from segmented image  
  5.2. Localisation + self-improving map  
  5.3. Path graph extraction from map


## 1.5. Business Risks

TBD

RI-1: The Cafeteria Employees Union might require that their contract be renegotiated to reflect the new employee roles and cafeteria hours of operation. (Probability = 0.6; Impact = 3)

RI-2: Too few employees might use the system, reducing the return on investment from the system development and the changes in cafeteria operating procedures. (Probability = 0.3; Impact = 9)

RI-3: Local restaurants might not agree to offer delivery, which would reduce employee satisfaction with the system and possibly their usage of it. (Probability = 0.3; Impact = 3)

RI-4: Sufficient delivery capacity might not be available, which means that employees would not always receive their meals on time and could not always request delivery for the desired times. (Probability = 0.5; Impact = 6).


# 2. Scope and Limitations


## 2.1. Major Features

TBD

FE-1: Order and pay for meals from the cafeteria menu to be picked up or delivered.

FE-2: Order and pay for meals from local restaurants to be delivered.

FE-3: Create, view, modify, and cancel meal subscriptions for standing or recurring meal orders, or for daily special meals.

FE-4: Create, view, modify, delete, and archive cafeteria menus.

FE-5: View ingredient lists and nutritional information for cafeteria menu items.

FE-6: Provide system access through corporate intranet, smartphone, tablet, and outside Internet access by authorized employees


## 2.2. Stakeholders profiles

TBD

| **Stakeholder** | **Major Value** | **Attitudes** | **Major Interests** | **Constraints** |
| --- | --- | --- | --- | --- |
| Corporate Management | Improved employee productivity; cost savings for cafeteria | Strong commitment through release 2; support for release 3 contingent on earlier results | Cost and employee time savings must exceed development and usage costs | None identified |
| Cafeteria Staff | More efficient use of staff time throughout the day; higher customer satisfaction | Concern about union relationships and possible downsizing; otherwise receptive | Job preservation | Training for staff in Internet usage needed; delivery staff and vehicles needed |
| Patrons | Better food selection; time savings; convenience | Strong enthusiasm, but might not use it as much as expected because of social value of eating lunches in cafeteria and restaurants | Simplicity of use; reliability of delivery; availability of food choices | Corporate intranet access, Internet access, or a mobile device is needed |
| PayrollDepartment | No benefit; needs to set up payroll deduction registration scheme | Not happy about the software work needed, but recognizes the value to the company and employees | Minimal changes in current payroll applications | No resources yet committed to make software changes |
| Restaurant Managers | Increased sales; marketing exposure to generate new customers | Receptive but cautious | Minimal new technology needed; concern about resources and costs of delivering meals | Might not have staff and capacity to handle order levels; might not have all menus online |


# 3. Software requirements specifications

## 3.1. Users and Characteristics

**TBD**

**Patron (favored)**

A Patron is a Process Impact employee who wants to order meals to be delivered from the company cafeteria. There are about 600 potential Patrons, of which 300 are expected to use the COS an average of 5 times per week each. Patrons will sometimes order multiple meals for group events or guests. An estimated 60 percent of orders will be placed using the corporate Intranet, with 40 percent of orders being placed from home or by smartphone or tablet apps.

**Cafeteria Staff**

The Process Impact cafeteria employs about 20 Cafeteria Staff, who will receive orders from the COS, prepare meals, package them for delivery, and request delivery. Most of the Cafeteria Staff will need training in the use of the hardware and software for the COS.

**Menu Manager**

The Menu Manager is a cafeteria employee who establishes and maintains daily menus of the food items available from the cafeteria. Some menu items may not be available for delivery. The Menu Manager will also define the cafeteria&#39;s daily specials. The Menu Manager will need to edit existing menus periodically.

**Meal Deliverer**

As the Cafeteria Staff prepare orders for delivery, they will issue delivery requests to a Meal Deliverer&#39;s smartphone. The Meal Deliverer will pick up the food and deliver it to the Patron. A Meal Deliverer&#39;s other interactions with the COS will be to confirm that a meal was (or was not) delivered.

## 3.2. Operating Environment Constraints

TBD

OE-1: The COS shall operate correctly with the following web browsers: Windows Internet Explorer versions 7, 8, and 9; Firefox versions 12 through 26; Google Chrome (all versions); and Apple Safari versions 4.0 through 8.0.

OE-2: The COS shall operate on a server running the current corporate-approved versions of Red Hat Linux and Apache HTTP Server.

OE-3: The COS shall permit user access from the corporate Intranet, from a VPN Internet connection, and by Android, iOS, and Windows smartphones and tablets.

## 3.3. Design and Implementation Constraints

TBD

CO-1: The system&#39;s design, code, and maintenance documentation shall conform to the _Process Impact Intranet Development Standard, Version 1.3_ [2].

CO-2: The system shall use the current corporate standard Oracle database engine.

CO-3: All HTML code shall conform to the HTML 5.0 standard.


## 3.4. Assumptions

TBD

- The cafeteria is open for breakfast, lunch, and supper every company business day in which employees are expected to be on site.

- The operation of the COS depends on changes being made in the Payroll System to accept payment requests for meals ordered with the COS.

- The operation of the COS depends on changes being made in the Cafeteria Inventory System to update the availability of food items as COS accepts meal orders.
- If a restaurant has its own on-line ordering system, the Cafeteria Ordering System must be able to communicate with it bi-directionally.

- Some food items that are available from the cafeteria will not be suitable for delivery, so the menus available to patrons of the COS must be a subset of the full cafeteria menus.

- The COS shall be used only for the cafeteria at the Process Impact campus people.

# 4. External Interface Requirements

## 4.1. User Interfaces

TBD

UI-1: The Cafeteria Ordering System screen displays shall conform to the _Process Impact Internet Application User Interface Standard, Version 2.0_ [3].

UI-2: The system shall provide a help link from each displayed webpage to explain how to use that page.

UI-3: The webpages shall permit complete navigation and food item selection by using the keyboard alone, in addition to using mouse and keyboard combinations.


## 4.2. Software Interfaces

TBD

SI-1: Cafeteria Inventory System

SI-1.1: The COS shall transmit the quantities of food items ordered to the Cafeteria Inventory System through a programmatic interface.

SI-1.2: The COS shall poll the Cafeteria Inventory System to determine whether a requested food item is available.

SI-1.3: When the Cafeteria Inventory System notifies the COS that a specific food item is no longer available, the COS shall remove that food item from the menu for the current date.

SI-2: Payroll System

The COS shall communicate with the Payroll System through a programmatic interface for the following operations:

SI-2.1: To allow a Patron to register and unregister for payroll deduction.

SI-2.2: To inquire whether a Patron is registered for payroll deduction.

SI-2.3: To inquire whether a Patron is eligible to register for payroll deduction.

SI-2.4: To submit a payment request for a purchased meal.

SI-2.5: To reverse all or part of a previous charge because a patron rejected a meal or wasn&#39;t satisfied with it, or because the meal was not delivered per the confirmed delivery instructions.


## 4.3. Communications Interfaces

CI-1: The COS shall send an email or text message (based on user account settings) to the Patron to confirm acceptance of an order, price, and delivery instructions.

CI-2: The COS shall send an email or text message (based on user account settings) to the Patron to report any problems with the meal order or delivery.