from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .policy_engine import evaluate_policy
from .routing import get_department_for_issue, get_default_severity, load_taxonomy
from .preprocessing import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "synthetic_feedback.csv"

REGIONS = ["Hyderabad", "Visakhapatnam", "Chennai", "Pune", "Mumbai", "Bengaluru"]
CHANNELS = ["Online Review", "In-app Rating", "Store Staff Note", "Call Centre"]
LANGUAGES = ["English", "Hinglish", "Telugu Transliteration", "Tamil Transliteration", "Marathi Transliteration"]
STORE_SIZES = ["Small", "Medium", "Large"]

CATEGORY_PRIORITY = [
    "Refrigeration Failure",
    "Hygiene",
    "Allergen Label Error",
    "Stock Shortage",
    "Delivery Delay",
    "Incorrect Price Label",
    "App Payment Failure",
    "Staff Misconduct",
    "Checkout Delay",
    "Store Cleanliness",
    "Food Contamination",
    "Product Damage",
    "Shelf Tag Missing",
    "Equipment Failure",
    "Freshness Concern",
    "Customer Service Delay",
    "Lighting Failure",
    "Security Concern",
    "Promotion Mismatch",
    "Billing Issue",
    "App Login Failure",
    "Queue Management",
    "Inventory Mismatch",
    "Packaging Damage",
    "Fresh Produce Quality",
    "Return Issue",
    "Parking Issue",
    "Unsafe Product",
    "Harassment",
    "Assortment Gap",
    "Order Cancellation",
    "Staff Courtesy",
    "Customer Praise",
]

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Refrigeration Failure": {
        "English": [
            "The freezer is not working and milk packets are spoiling.",
            "Fridge temperature is too high; the frozen section is warm.",
            "Cold storage failed again and products are getting spoiled.",
        ],
        "Hinglish": [
            "Store lo freezer work avvatledu, milk packets spoil avtunnayi!",
            "Fridge temperature chala ekkuva undi, ice cream melt avtundi.",
            "Freezer band ayindi, items spoil avuthunnayi.",
        ],
        "Telugu Transliteration": [
            "Freezer pani cheyatledu, milk packets spoil avtunnayi.",
            "Cold room lo temperature ekkuva undi, items spoil avtunnayi.",
        ],
        "Tamil Transliteration": [
            "Freezer seri illa, milk packets spoil aaguthu.",
            "Cold storage work aagala, goods spoil aaguthu.",
        ],
        "Marathi Transliteration": [
            "Freezer chalat nahi aahe, dudh packets spoil hot aahet.",
            "Cold storage madhye problem aahe ani mal spoil hotoy.",
        ],
    },
    "Hygiene": {
        "English": [
            "The washroom and aisle are dirty and smell bad.",
            "Hygiene is poor; spills were not cleaned on time.",
            "Cockroaches near the bakery area and unclean shelves.",
        ],
        "Hinglish": [
            "Store bahut dirty hai, smell aa rahi hai aur cleaning nahi hui.",
            "Washroom ganda hai aur floor pe spill pada hai.",
            "Safai theek nahi hai, cockroach dikha.",
        ],
        "Telugu Transliteration": [
            "Store lo cleanliness ledu, smell vastondi.",
            "Washroom chala dirty ga undi, spill clean cheyatledu.",
        ],
        "Tamil Transliteration": [
            "Store romba dirty irukku, smell varudhu.",
            "Washroom clean aagala, floor spill irukku.",
        ],
        "Marathi Transliteration": [
            "Store khup dirty aahe, smell yetey.",
            "Washroom ani floor var safai nahi zali.",
        ],
    },
    "Allergen Label Error": {
        "English": [
            "Incorrect allergen information printed on the food packet.",
            "The label says no nuts, but the ingredient list includes nuts.",
            "Allergen warning missing from the packaged item.",
        ],
        "Hinglish": [
            "Wrong allergen label on the packet, very risky.",
            "Food packet pe allergen info galat hai.",
            "Pack label me nuts mention nahi hai but andar hai.",
        ],
        "Telugu Transliteration": [
            "Food packet me allergen label tappu ga undi.",
            "Nuts unnayi kani label lo mention cheyaledu.",
        ],
        "Tamil Transliteration": [
            "Food packet la allergen label thappu ah irukku.",
            "Nuts irukku aana label sollala.",
        ],
        "Marathi Transliteration": [
            "Food packet var allergen label chukicha aahe.",
            "Nuts aahet pan label var dile nahi.",
        ],
    },
    "Stock Shortage": {
        "English": ["The shelf is empty and the product is out of stock.", "Stock shortage in the dairy aisle."],
        "Hinglish": ["Shelf khali hai, stock nahi hai.", "Dairy section me stock shortage hai."],
        "Telugu Transliteration": ["Shelf empty undi, stock ledu.", "Dairy lo stock shortage undi."],
        "Tamil Transliteration": ["Shelf la stock illa.", "Dairy section la stock shortage irukku."],
        "Marathi Transliteration": ["Shelf rikami aahe, stock nahi aahe.", "Dairy madhye stock shortage aahe."],
    },
    "Delivery Delay": {
        "English": ["Delivery arrived late and the fresh stock missed peak hours.", "Truck delay caused late replenishment."],
        "Hinglish": ["Delivery late aayi, replenish time miss ho gaya.", "Truck late tha aur stock time pe nahi aaya."],
        "Telugu Transliteration": ["Delivery late ga vachindi.", "Truck delay valla stock late ga refill ayindi."],
        "Tamil Transliteration": ["Delivery late ah vandhuchu.", "Truck delay nala stock late refill aachu."],
        "Marathi Transliteration": ["Delivery late ali.", "Truck delay mule stock late ala."],
    },
    "Incorrect Price Label": {
        "English": ["The price label is incorrect and the billing amount is wrong.", "MRP on shelf does not match billing."],
        "Hinglish": ["Price tag galat hai, billing bhi mismatch hai.", "Shelf label aur bill match nahi kar raha."],
        "Telugu Transliteration": ["Price label tappu undi.", "Billing amount shelf price tho match avvatledu."],
        "Tamil Transliteration": ["Price label thappu ah irukku.", "Bill amount shelf price kooda match aagala."],
        "Marathi Transliteration": ["Price tag chukicha aahe.", "Bill ani shelf price match hot nahi."],
    },
    "App Payment Failure": {
        "English": ["Payment keeps failing in the app checkout flow.", "UPI payment failed multiple times."],
        "Hinglish": ["App payment fail ho raha hai.", "Checkout pe transaction decline ho gaya."],
        "Telugu Transliteration": ["App payment fail avtundi.", "UPI payment work avvatledu."],
        "Tamil Transliteration": ["App payment fail aaguthu.", "Checkout la payment work aagala."],
        "Marathi Transliteration": ["App payment fail hot aahe.", "UPI payment work hot nahi."],
    },
    "Staff Misconduct": {
        "English": ["A staff member behaved inappropriately with a customer.", "The employee was rude and threatening."],
        "Hinglish": ["Staff member ne inappropriate behavior kiya.", "Employee rude tha aur customer ko threaten kiya."],
        "Telugu Transliteration": ["Staff member inappropriate ga behave chesaru.", "Customer tho rude ga matladaru."],
        "Tamil Transliteration": ["Staff member inappropriate ah nadandhukitanga.", "Customer kitta rude ah pesinaanga."],
        "Marathi Transliteration": ["Staff member ne inappropriate vatanuk keli.", "Customer shi rude vaatla."],
    },
    "Checkout Delay": {
        "English": ["Checkout lines are too long during evening peak hours.", "Waiting time at billing counter is high."],
        "Hinglish": ["Checkout queue bahut lambi hai.", "Billing counter pe wait zyada hai."],
        "Telugu Transliteration": ["Billing queue chala pedda ga undi.", "Checkout lo delay avtundi."],
        "Tamil Transliteration": ["Checkout queue romba perusa irukku.", "Billing counter la wait adhigama irukku."],
        "Marathi Transliteration": ["Checkout line khup mothi aahe.", "Billing counter var wait jasta aahe."],
    },
    "Store Cleanliness": {
        "English": ["The store floor is dirty and trash is visible near the entrance.", "Cleaning was missed in the aisles."],
        "Hinglish": ["Store floor dirty hai aur trash pada hai.", "Aisle ki cleaning nahi hui."],
        "Telugu Transliteration": ["Store floor dirty ga undi.", "Entrance daggara trash undi."],
        "Tamil Transliteration": ["Store floor dirty ah irukku.", "Entrance pakkathula trash irukku."],
        "Marathi Transliteration": ["Store floor dirty aahe.", "Entrance jawal kachra aahe."],
    },
    "Customer Praise": {
        "English": ["Great shopping experience and helpful staff.", "Thanks, the store was clean and well organized."],
        "Hinglish": ["Great experience, staff helpful tha.", "Store clean tha aur shopping smooth thi."],
        "Telugu Transliteration": ["Shopping experience chala bagundi.", "Staff helpful ga unnaru."],
        "Tamil Transliteration": ["Shopping experience romba nalla irundhuchu.", "Staff helpful ah irundhaanga."],
        "Marathi Transliteration": ["Shopping experience khup changla hota.", "Staff helpful hote."],
    },
    "Food Contamination": {
        "English": ["Found fungus on the packaged bread, it looks contaminated.", "Mold growth inside the container, food poisoning risk."],
        "Hinglish": ["Bread pe fungus lag gaya hai, contaminated lag raha hai.", "Container ke andar mould hai, food poisoning ka risk hai."],
        "Telugu Transliteration": ["Bread me fungus undi, contaminated ga undi.", "Container lo mold undi, food poisoning risk undi."],
        "Tamil Transliteration": ["Bread la fungus irundhu, contaminated ah irukku.", "Container la mold undi, food poisoning risk undi."],
        "Marathi Transliteration": ["Bread var fungus aahe, contaminated aahe.", "Container aatil mold aahe, food poisoning cha risk aahe."],
    },
    "Unsafe Product": {
        "English": ["Broken bottle with sharp glass on the floor, it is dangerous.", "A leaking item near the electrical socket is a hazard."],
        "Hinglish": ["Floor pe sharp glass pada hai, dangerous situation hai.", "Electrical socket ke paas leaking item hai, hazard hai."],
        "Telugu Transliteration": ["Floor me sharp glass undi, dangerous ga undi.", "Electrical socket daggara leaking item undi, hazard undi."],
        "Tamil Transliteration": ["Floor la sharp glass irukku, dangerous ah irukku.", "Electrical socket pakkathula leaking item irukku, hazard irukku."],
        "Marathi Transliteration": ["Floor var sharp glass aahe, dangerous aahe.", "Electrical socket jwalil leaking item aahe, hazard aahe."],
    },
    "Harassment": {
        "English": ["A staff member was harassing and following me in the aisle.", "Unsafe behaviour from a staff member, it was harassment."],
        "Hinglish": ["Staff member ne harassment kiya aur mere peeche aaya.", "Staff se unsafe behaviour tha, ye harassment hai."],
        "Telugu Transliteration": ["Staff member harassment chesaru, naa meeda follow chesaru.", "Staff nundi unsafe behaviour undi, ee harassment."],
        "Tamil Transliteration": ["Staff member harassment cheythu, naa pinadi follow cheshthu.", "Staff kitta unsafe behaviour irundhu, idhu harassment."],
        "Marathi Transliteration": ["Staff member ne harassment keli ani maza pachha aala.", "Staff kade unsafe behaviour hota, he harassment aahe."],
    },
    "Security Concern": {
        "English": ["Someone tried to walk out with unpaid items, security needs attention.", "The area behind the store felt unsafe, a fight started nearby."],
        "Hinglish": ["Bina bill ke koi bahar ja raha tha, security ko dekhna chahiye.", "Store ke peeche area unsafe tha, wahan fight start hui."],
        "Telugu Transliteration": ["Bill leni kodiga bahar vellipoyindi, security ni choostavali.", "Store pakkathula area unsafe ga undi, akkada fight start ayindi."],
        "Tamil Transliteration": ["Bill illa oruvanu veliya vanakkum, security ni paatalama venum.", "Store pinbathula area unsafe ah irundhu, inga fight start aayidhu."],
        "Marathi Transliteration": ["Billa shivay koni baher gela, security ne baghaycha pahije.", "Store maghe area unsafe hota, tithhe fight zali."],
    },
    "Freshness Concern": {
        "English": ["The vegetables are not fresh and smell spoiled.", "Stale bread on the shelf, the freshness date has passed."],
        "Hinglish": ["Sabzi fresh nahi hai aur smell kharab aa rahi hai.", "Shelf pe stale bread hai, freshness date nikal gaya hai."],
        "Telugu Transliteration": ["Sabzi fresh nahi undi, smell kharab vastundi.", "Shelf lo stale bread undi, freshness date telipoyindi."],
        "Tamil Transliteration": ["Sabzi fresh illa, smell kharab varudhu.", "Shelf la stale bread irukku, freshness date selipoyidhu."],
        "Marathi Transliteration": ["Sabzi fresh nahi aahe, smell kharab yetey.", "Shelf var stale bread aahe, freshness date seniority aahe."],
    },
    "Equipment Failure": {
        "English": ["The billing scanner is broken and not working at all.", "The printer machine is kaput, we could not print the bill."],
        "Hinglish": ["Billing scanner broken hai aur work nahi kar raha.", "Printer machine kaput hai, bill print nahi hua."],
        "Telugu Transliteration": ["Billing scanner broken undi, work avvatledu.", "Printer machine kaput undi, bill print avvatledu."],
        "Tamil Transliteration": ["Billing scanner broken ah irukku, work aagala.", "Printer machine kaput ah irukku, bill print aagala."],
        "Marathi Transliteration": ["Billing scanner broken aahe ani work hot nahi.", "Printer machine kaput aahe, bill print zala nahi."],
    },
    "Shelf Tag Missing": {
        "English": ["The product has no shelf tag and the price tag is absent.", "Missing tag on the shelf, customers are confused."],
        "Hinglish": ["Product ka shelf tag nahi hai, price tag absent hai.", "Shelf pe tag missing hai, customers confuse ho rahe hain."],
        "Telugu Transliteration": ["Product ki shelf tag ledu, price tag absent undi.", "Shelf lo tag missing undi, customers confuse avtunnaru."],
        "Tamil Transliteration": ["Product ku shelf tag illa, price tag absent ah irukku.", "Shelf la tag missing ah irukku, customers confuse aagarnthu."],
        "Marathi Transliteration": ["Product cha shelf tag nahi, price tag absent aahe.", "Shelf var tag missing aahe, customers confuse hot aahet."],
    },
    "Fresh Produce Quality": {
        "English": ["The fresh produce section has rotten fruit and soft bananas.", "Vegetables are bad quality, the produce was not fresh."],
        "Hinglish": ["Fresh produce section me fruit rotten hai aur bananas soft hain.", "Sabzi quality kharab hai, produce fresh nahi tha."],
        "Telugu Transliteration": ["Fresh produce section lo fruit rotten undi, bananas soft unnayi.", "Sabzi quality kharab undi, produce fresh kaadu."],
        "Tamil Transliteration": ["Fresh produce section la fruit rotten irukku, bananas soft unnai.", "Sabzi quality kharab irukku, produce fresh illa."],
        "Marathi Transliteration": ["Fresh produce section madhe fruit rotten aahe, bananas soft aahet.", "Sabzi quality kharab aahe, produce fresh navhati."],
    },
    "Packaging Damage": {
        "English": ["The packaging is torn and the pack is leaking on the shelf.", "A crushed pack arrived with damaged packaging."],
        "Hinglish": ["Packaging torn hai aur pack shelf pe leak ho raha hai.", "Crushed pack aaya tha, packaging damaged thi."],
        "Telugu Transliteration": ["Packaging torn undi, pack shelf me leak avtundi.", "Crushed pack vachindi, packaging damaged undi."],
        "Tamil Transliteration": ["Packaging torn irukku, pack shelf la leak aaguthu.", "Crushed pack vandhudhu, packaging damaged irundhu."],
        "Marathi Transliteration": ["Packaging torn aahe ani pack shelf var leak hot aahe.", "Crushed pack aala, packaging damaged hoti."],
    },
    "Product Damage": {
        "English": ["The damaged product I bought was cracked and leaking.", "Broken item in the aisle, the product was damaged."],
        "Hinglish": ["Jo damaged product liya tha wo cracked aur leaking tha.", "Aisle me broken item tha, product damaged tha."],
        "Telugu Transliteration": ["Damaged product kondaanu adi cracked aur leaking undi.", "Aisle lo broken item undi, product damaged undi."],
        "Tamil Transliteration": ["Damaged product kondaana vachi adhigama cracked ah leaking irundhu.", "Aisle la broken item irundhu, product damaged irundhu."],
        "Marathi Transliteration": ["Damaged product ghetla te cracked ani leaking hota.", "Aisle madhe broken item hota, product damaged hota."],
    },
    "Inventory Mismatch": {
        "English": ["Inventory mismatch at the warehouse, system stock differs from physical stock.", "The counts are wrong and the barcode mismatch is visible."],
        "Hinglish": ["Warehouse me inventory mismatch hai, system stock aur physical stock alag hai.", "Counts galat hain aur barcode mismatch dikh raha hai."],
        "Telugu Transliteration": ["Warehouse lo inventory mismatch undi, system stock physical stock ki vere.", "Counts tappu unnayi, barcodeMismatch kaadu avtundi."],
        "Tamil Transliteration": ["Warehouse la inventory mismatch irukku, system stock physical stock vera irukku.", "Counts thappu irunnu, barcode mismatch kaadu aayidhu."],
        "Marathi Transliteration": ["Warehouse madhe inventory mismatch aahe, system stock ani physical stock vegla aahet.", "Counts chukicha aahet ani barcodeMismatch dakhavtoy."],
    },
    "Promotion Mismatch": {
        "English": ["The offer was not applied and the discount is missing on the bill.", "Coupon promo did not apply even though it was eligible."],
        "Hinglish": ["Offer apply nahi hua aur bill pe discount missing hai.", "Coupon promo apply nahi hua jabki eligible tha."],
        "Telugu Transliteration": ["Offer apply avvatledu, bill me discount missing undi.", "Coupon promo apply avvatledu, eligible ayina."],
        "Tamil Transliteration": ["Offer apply aagala, bill la discount missing irukku.", "Coupon promo apply aagala eligible irundhu."],
        "Marathi Transliteration": ["Offer apply zala nahi ani bill var discount missing aahe.", "Coupon promo apply zala nahi even though eligible hota."],
    },
    "Billing Issue": {
        "English": ["The bill is wrong and I was charged twice for the same item.", "Invoice amount is incorrect, the receipt does not match."],
        "Hinglish": ["Bill galat hai aur same item ke liye do baar charge hua.", "Invoice amount incorrect hai, receipt match nahi karta."],
        "Telugu Transliteration": ["Bill tappu undi, same item ki dubbara charge ayindi.", "Invoice amount tappu undi, receipt match avvatledu."],
        "Tamil Transliteration": ["Bill thappu irukku, same item ku irandu vaati charge aayidhu.", "Invoice amount thappu irukku, receipt match aagala."],
        "Marathi Transliteration": ["Bill chukicha aahe ani same item sathi don vaat charge zaale.", "Invoice amount chukicha aahe, receipt match hot nahi."],
    },
    "App Login Failure": {
        "English": ["The app login failed and I cannot sign in to my account.", "The app crash keeps happening and the otp is not coming."],
        "Hinglish": ["App login failed ho raha hai, account me sign in nahi ho raha.", "App crash hota rehta hai aur otp nahi aa raha."],
        "Telugu Transliteration": ["App login fail avtundi, account lo sign in avvatledu.", "App crash avtundi, otp raaledu."],
        "Tamil Transliteration": ["App login fail aaguthu, account la sign in aagala.", "App crash aagum ma, otp varudhu illa."],
        "Marathi Transliteration": ["App login fail hot aahe, account madhe sign in hot nahi.", "App crash kahi hot aahe ani otp yet nahi."],
    },
    "Return Issue": {
        "English": ["My return was rejected and the refund is still pending.", "The return problem has not been resolved, exchange was refused."],
        "Hinglish": ["Mera return reject ho gaya aur refund abhi bhi pending hai.", "Return problem solve nahi hua, exchange refuse kar diya."],
        "Telugu Transliteration": ["Nenu return reject chesaru, refund pendu ga undi.", "Return problem solve avvatledu, exchange refuse chesaru."],
        "Tamil Transliteration": ["Enakku return reject aayidhu, refund pendu irukku.", "Return problem solve aagala, exchange refuse aayidhu."],
        "Marathi Transliteration": ["Maza return reject zala ani refund ajun pending aahe.", "Return problem solve zala nahi, exchange refuse karun diye."],
    },
    "Queue Management": {
        "English": ["The queue management at billing is poor and I was waiting for a long time.", "Long waiting line and no queue control at the entrance."],
        "Hinglish": ["Billing pe queue management kharab hai aur main bahut der se wait kar raha tha.", "Entrance pe lamba waiting line hai aur queue control nahi hai."],
        "Telugu Transliteration": ["Billing me queue management kharab undi, bahut neru wait chesa.", "Entrance lo pedda waiting line undi, queue control ledu."],
        "Tamil Transliteration": ["Billing la queue management kharab irukku, romba neru wait chesa.", "Entrance la perusa waiting line irukku, queue control illa."],
        "Marathi Transliteration": ["Billing var queue management kharab aahe ani mi khup vel wait kela.", "Entrance var lambi waiting line aahe, queue control nahi."],
    },
    "Lighting Failure": {
        "English": ["The lights are off in the back aisle and the area is dark.", "There is a lighting issue, the bulb is not working."],
        "Hinglish": ["Peeche wali aisle me lights off hain aur area dark hai.", "Lighting issue hai, bulb work nahi kar raha."],
        "Telugu Transliteration": ["Peeche aisle lo lights off unnayi, area dark undi.", "Lighting issue undi, bulb work avvatledu."],
        "Tamil Transliteration": ["Pinadi aisle la lights off irunnu, area dark irukku.", "Lighting issue irukku, bulb work aagala."],
        "Marathi Transliteration": ["Maghya aisle var lights off aahet ani area dark aahe.", "Lighting issue aahe, bulb work hot nahi."],
    },
    "Parking Issue": {
        "English": ["The parking lot is full and the entrance is blocked.", "No parking space near the store and the vehicle is parked badly."],
        "Hinglish": ["Parking lot full hai aur entrance blocked hai.", "Store ke paas parking space nahi hai aur vehicle galat park kiya hai."],
        "Telugu Transliteration": ["Parking lot full undi, entrance blocked undi.", "Store daggara parking space ledu, vehicle sarva incorrect ga undi."],
        "Tamil Transliteration": ["Parking lot full irukku, entrance blocked irukku.", "Store pakkathula parking space illa, vehicle thappu vaadi park cheshthu."],
        "Marathi Transliteration": ["Parking lot full aahe ani entrance blocked aahe.", "Store javal parking space nahi ani vehicle chukicha park keli aahe."],
    },
    "Customer Service Delay": {
        "English": ["There was no response for a long time and I was waiting for help.", "Staff were unavailable at the service desk, service delay."],
        "Hinglish": ["Bahut der se koi response nahi mila aur main help ka wait kar raha tha.", "Service desk pe staff available nahi the, service delay."],
        "Telugu Transliteration": ["Bahut neru response ledu, help kosam wait chesa.", "Service desk me staff available kadu, service delay undi."],
        "Tamil Transliteration": ["Romba neru response illa, help kosam wait chesa.", "Service desk la staff available illa, service delay undi."],
        "Marathi Transliteration": ["Khup vel response nahi aala ani mi help cha wait kela.", "Service desk var staff available navhati, service delay hota."],
    },
    "Assortment Gap": {
        "English": ["There is no variety in this section, the assortment is very limited selection.", "No options for the products I need, the shelf has missing variety."],
        "Hinglish": ["Is section me variety nahi hai, assortment bahut kam hai.", "Jo chahiye uska koi option nahi, shelf par missing variety hai."],
        "Telugu Transliteration": ["E section lo variety ledu, assortment chala kam undi.", "Kavalsalaniki option ledu, shelf me missing variety undi."],
        "Tamil Transliteration": ["E section la variety illa, assortment romba kam irukku.", "Theevatthaanaaga option illa, shelf la missing variety irukku."],
        "Marathi Transliteration": ["Aa section madhe variety nahi aahe, assortment khup kam aahe.", "Je lagte te option nahi, shelf var missing variety aahe."],
    },
    "Order Cancellation": {
        "English": ["My app order was cancelled without any reason and the refund has not come.", "The order was cancelled by the delivery app and the refund is still pending."],
        "Hinglish": ["Mera app order bina reason cancel ho gaya aur refund nahi aaya.", "Delivery app ne order cancel kiya aur refund pending hai."],
        "Telugu Transliteration": ["Nenu app order kannada reason ledu cancel ayindi, refund radu.", "Delivery app order ni cancel chesindi, refund pendu undi."],
        "Tamil Transliteration": ["Enakku app order karuvana cancel aayidhu, refund varalillla.", "Delivery app order cancel cheshthu, refund pendu irukku."],
        "Marathi Transliteration": ["Maza app order karma cancel zala, refund nahi aala.", "Delivery app ne order cancel keli, refund ajun pending aahe."],
    },
    "Staff Courtesy": {
        "English": ["The staff member was rude and had a bad attitude at the counter.", "An uncourteous employee was not polite with the customer."],
        "Hinglish": ["Staff member rude tha aur counter pe uska attitude kharab tha.", "Ek uncourteous employee customer ke saath polite nahi tha."],
        "Telugu Transliteration": ["Staff member rude ga undi, counter me attitude kharab undi.", "Uncourteous employee customer tho polite ga undi kaadu."],
        "Tamil Transliteration": ["Staff member rude ah irundhu, counter la attitude kharab irundhu.", "Uncourteous employee customer kitta polite ah illa."],
        "Marathi Transliteration": ["Staff member rude hota ani counter var attitude kharab hota.", "Uncourteous employee customer shi polite nahi hota."],
    },
}

INCIDENT_PROFILES = [
    {
        "incident_tag": "refrigeration_spike_st042",
        "store_id": "ST-042",
        "region": "Visakhapatnam",
        "issue_category": "Refrigeration Failure",
        "start_offset_weeks": 6,
        "weekly_counts": [1, 2, 3, 5, 7, 9],
        "known_escalation_offset_weeks": 9,
    },
    {
        "incident_tag": "hygiene_spike_st007",
        "store_id": "ST-007",
        "region": "Hyderabad",
        "issue_category": "Hygiene",
        "start_offset_weeks": 8,
        "weekly_counts": [1, 1, 2, 3, 4, 4, 5, 6],
        "known_escalation_offset_weeks": 11,
    },
    {
        "incident_tag": "allergen_spike_st019",
        "store_id": "ST-019",
        "region": "Pune",
        "issue_category": "Allergen Label Error",
        "start_offset_weeks": 4,
        "weekly_counts": [1, 1, 2, 2, 3, 4],
        "known_escalation_offset_weeks": 7,
    },
    {
        "incident_tag": "low_volume_store_st031",
        "store_id": "ST-031",
        "region": "Bengaluru",
        "issue_category": "Stock Shortage",
        "start_offset_weeks": 5,
        "weekly_counts": [1, 2, 2, 3, 3],
        "known_escalation_offset_weeks": 8,
    },
]


@dataclass(slots=True)
class SyntheticDatasetBundle:
    dataframe: pd.DataFrame
    incident_profiles: list[dict]
    path: Path



def _build_store_master() -> pd.DataFrame:
    rows = []
    store_idx = 1
    region_cycle = REGIONS * 9
    for region in region_cycle[:50]:
        size = np.random.choice(STORE_SIZES, p=[0.35, 0.4, 0.25])
        rows.append(
            {
                "store_id": f"ST-{store_idx:03d}",
                "region": region,
                "store_size": size,
                "store_name": f"Store {store_idx:03d}",
            }
        )
        store_idx += 1
    return pd.DataFrame(rows)


def _choose_language(rng: np.random.Generator) -> str:
    return rng.choice(LANGUAGES, p=[0.46, 0.2, 0.14, 0.1, 0.1])


def _template_for(category: str, language: str, rng: np.random.Generator) -> str:
    category_templates = TEMPLATES.get(category)
    if category_templates is None:
        # Falling back to another category's text silently corrupts the ground
        # truth, so fail loudly instead of generating mislabelled records.
        raise KeyError(
            f"No templates defined for category {category!r}. "
            "Add multilingual templates before generating this category."
        )
    options = category_templates.get(language) or category_templates.get("English") or [f"{category} reported."]
    text = str(rng.choice(options))
    noise_suffixes = ["", " 😟", " !!", " pls check", " urgent", " asap"]
    return text + str(rng.choice(noise_suffixes))


def _base_category_weights() -> dict[str, float]:
    weights = {
        "Customer Praise": 0.12,
        "Customer Service Delay": 0.08,
        "Store Cleanliness": 0.08,
        "Incorrect Price Label": 0.07,
        "Stock Shortage": 0.07,
        "Checkout Delay": 0.06,
        "Delivery Delay": 0.05,
        "App Payment Failure": 0.05,
        "Equipment Failure": 0.05,
        "Shelf Tag Missing": 0.05,
        "Hygiene": 0.05,
        "Fresh Produce Quality": 0.04,
        "Packaging Damage": 0.04,
        "Billing Issue": 0.04,
        "Inventory Mismatch": 0.04,
        "Promotion Mismatch": 0.03,
        "App Login Failure": 0.03,
        "Store Cleanliness": 0.03,
        "Product Damage": 0.03,
        "Freshness Concern": 0.03,
        "Refrigeration Failure": 0.02,
        "Allergen Label Error": 0.02,
        "Food Contamination": 0.02,
        "Unsafe Product": 0.02,
        "Security Concern": 0.02,
        "Staff Misconduct": 0.02,
        "Harassment": 0.01,
        "Queue Management": 0.01,
        "Lighting Failure": 0.01,
        "Parking Issue": 0.01,
        "Return Issue": 0.01,
        "Assortment Gap": 0.01,
        "Order Cancellation": 0.01,
        "Staff Courtesy": 0.01,
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


BASE_WEIGHTS = _base_category_weights()


def _sample_category(rng: np.random.Generator, weights: dict[str, float] | None = None) -> str:
    categories = list((weights or BASE_WEIGHTS).keys())
    probs = list((weights or BASE_WEIGHTS).values())
    return str(rng.choice(categories, p=probs))


def _build_weekly_schedule(store_master: pd.DataFrame, weeks: int, rng: np.random.Generator) -> list[dict]:
    base_end = pd.Timestamp.today().normalize().to_period("W-MON").start_time
    week_starts = [base_end - pd.Timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]
    schedule: list[dict] = []
    for _, store in store_master.iterrows():
        size = store["store_size"]
        if size == "Small":
            low, high = 8, 13
        elif size == "Large":
            low, high = 24, 36
        else:
            low, high = 14, 24
        for week_start in week_starts:
            base_volume = int(rng.integers(low, high + 1))
            schedule.append(
                {
                    "store_id": store["store_id"],
                    "region": store["region"],
                    "store_size": size,
                    "week_start": week_start,
                    "base_volume": base_volume,
                }
            )
    return schedule


def _incident_targets(week_start: pd.Timestamp) -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    for profile in INCIDENT_PROFILES:
        start_week = week_start + pd.Timedelta(weeks=profile["start_offset_weeks"])
        if week_start >= start_week:
            offset = int((week_start - start_week) / pd.Timedelta(weeks=1))
            counts = profile["weekly_counts"]
            if 0 <= offset < len(counts):
                targets[(profile["store_id"], profile["issue_category"])] = counts[offset]
    return targets


def _incident_profile_lookup() -> dict[tuple[str, str], dict]:
    return {(item["store_id"], item["issue_category"]): item for item in INCIDENT_PROFILES}


def generate_synthetic_dataset(seed: int = 2026, weeks: int = 12) -> SyntheticDatasetBundle:
    rng = np.random.default_rng(seed)
    store_master = _build_store_master()
    schedule = _build_weekly_schedule(store_master, weeks=weeks, rng=rng)
    profile_lookup = _incident_profile_lookup()
    rows: list[dict] = []
    feedback_id = 1

    for item in schedule:
        store_id = item["store_id"]
        region = item["region"]
        week_start = pd.Timestamp(item["week_start"])
        base_volume = int(item["base_volume"])
        store_rows: list[dict] = []

        # Determine if this store has a seeded incident active this week.
        active_incident = None
        for profile in INCIDENT_PROFILES:
            if profile["store_id"] == store_id:
                incident_start = pd.Timestamp.today().normalize().to_period("W-MON").start_time - pd.Timedelta(weeks=profile["start_offset_weeks"])
                if week_start >= incident_start:
                    offset = int((week_start - incident_start) / pd.Timedelta(weeks=1))
                    if 0 <= offset < len(profile["weekly_counts"]):
                        active_incident = {
                            "profile": profile,
                            "offset": offset,
                            "count": profile["weekly_counts"][offset],
                        }
                        break

        incident_count = active_incident["count"] if active_incident else 0
        total_feedback = base_volume + incident_count
        if total_feedback < 1:
            total_feedback = 1

        # General background distribution.
        categories = list(BASE_WEIGHTS.keys())
        weights = list(BASE_WEIGHTS.values())
        week_categories = rng.choice(categories, size=total_feedback, p=weights)

        # If an incident is active, overwrite some records with the incident category.
        if active_incident:
            incident_category = active_incident["profile"]["issue_category"]
            incident_injections = min(incident_count, total_feedback)
            if incident_injections > 0:
                inject_idx = rng.choice(np.arange(total_feedback), size=incident_injections, replace=False)
                for idx in inject_idx:
                    week_categories[idx] = incident_category

        for idx in range(total_feedback):
            category = str(week_categories[idx])
            language = _choose_language(rng)
            channel = str(rng.choice(CHANNELS))
            day_offset = int(rng.integers(0, 7))
            hour = int(rng.integers(7, 22))
            minute = int(rng.integers(0, 60))
            timestamp = week_start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)

            feedback_text = _template_for(category, language, rng)
            if active_incident and category == active_incident["profile"]["issue_category"]:
                feedback_text = _template_for(category, language, rng)
                if active_incident["offset"] >= 2 and idx % 2 == 0:
                    feedback_text += " repeated complaint from another channel"

            policy = evaluate_policy(feedback_text, category)
            department = get_department_for_issue(category, feedback_text)
            severity = policy.severity if policy.severity != "MEDIUM" else get_default_severity(category)
            actionable = bool(policy.actionable)

            incident_tag = ""
            known_incident_flag = False
            if active_incident and category == active_incident["profile"]["issue_category"]:
                incident_tag = active_incident["profile"]["incident_tag"]
                known_incident_flag = True

            store_rows.append(
                {
                    "feedback_id": f"FB-{feedback_id:06d}",
                    "store_id": store_id,
                    "region": region,
                    "store_size": item["store_size"],
                    "timestamp": timestamp,
                    "channel": channel,
                    "feedback_text": feedback_text,
                    "issue_category": category,
                    "severity": severity,
                    "actionable": actionable,
                    "department": department,
                    "dataset_mode": "Synthetic Demonstration Data",
                    "incident_tag": incident_tag,
                    "known_incident_flag": known_incident_flag,
                    "language_hint": language,
                    "feedback_text_normalized": normalize_text(feedback_text),
                }
            )
            feedback_id += 1

        rows.extend(store_rows)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "store_id", "feedback_id"]).reset_index(drop=True)
    df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
    df["store_week_id"] = df["store_id"] + "-" + df["week_start"].dt.strftime("%Y-%m-%d")
    df["week_label"] = df["week_start"].dt.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return SyntheticDatasetBundle(dataframe=df, incident_profiles=INCIDENT_PROFILES, path=CSV_PATH)


@np.vectorize
def _safe_str(value: object) -> str:
    return str(value)


@lru_cache(maxsize=1)
def load_or_create_synthetic_dataset() -> SyntheticDatasetBundle:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "week_start" not in df.columns and "timestamp" in df.columns:
            df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
        return SyntheticDatasetBundle(dataframe=df, incident_profiles=INCIDENT_PROFILES, path=CSV_PATH)
    return generate_synthetic_dataset()


def incident_profiles_dataframe() -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().to_period("W-MON").start_time
    rows = []
    for profile in INCIDENT_PROFILES:
        start_week = today - pd.Timedelta(weeks=profile["start_offset_weeks"])
        escalation_week = today - pd.Timedelta(weeks=profile["known_escalation_offset_weeks"])
        rows.append(
            {
                **profile,
                "incident_start_date": start_week,
                "known_escalation_date": escalation_week,
            }
        )
    return pd.DataFrame(rows)
