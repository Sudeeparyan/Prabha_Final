# EMG Data — Priya's Personal Memory Graph

This folder contains all knowledge nodes for the Editable Memory Graph (EMG).  
Edit the JSON files here — `app.py` loads them automatically on startup.

---

## Folder Structure

```
emg_data/
├── README.md              ← this file
├── all_nodes.json         ← MASTER FILE (auto-generated, loaded by app.py)
├── edges.json             ← directed edges between nodes
└── nodes/
    ├── faq.json           ← domain policy knowledge (12 nodes)
    ├── preference.json    ← Priya's personal settings (7 nodes)
    ├── event.json         ← live timestamped events (15 nodes)
    └── account_state.json ← current profile facts (5 nodes)
```

---

## Node Types (39 total, 38 edges)

### FAQ — Domain Policy Knowledge (12 nodes)
Static reference knowledge that rarely changes.

| ID | Label | What it covers |
|----|-------|---------------|
| faq_transfers | Bank Transfers FAQ | Transfer times, daily limits, SEPA |
| faq_credit | Credit Score FAQ | How to improve, utilisation rules |
| faq_fraud | Fraud Reporting FAQ | Freeze card, dispute process, temp credit |
| faq_travel | Travel Policy FAQ | Bank notification, insurance, carry-on |
| faq_hotel | Hotel Booking FAQ | Cancellation policy, check-in/out |
| faq_pto | PTO Policy FAQ | Notice period, 25-day allowance, rollover |
| faq_medication | Medication Guidance | Never skip, repeat prescriptions |
| faq_diet | Vegetarian GF Diet | Protein sources, fibre, meal ideas |
| faq_car | Car Maintenance FAQ | Oil change, NCT, tyre pressure |
| faq_capabilities | What Jarvis Can Do | Full capability list |
| faq_education | NCI Student Services | Library hours, databases, appeals |
| faq_fitness | Fitness Guidelines | Cardio, strength, recovery, hydration |

---

### Preference — Priya's Personal Settings (7 nodes)
User-specific preferences. These can be UPDATED via CRUD at any time.

| ID | Label | What it stores |
|----|-------|---------------|
| pref_dietary | Dietary Preferences | Vegetarian, GF, shellfish allergy, calorie target |
| pref_commute | Commute Preferences | Sandymount → Grand Canal Dock via DART, 8am |
| pref_notifications | Notification Prefs | Email preferred, quiet hours 10pm-7am |
| pref_music | Music Preferences | Lo-fi commute, jazz evenings, no heavy metal |
| pref_fitness | Fitness Preferences | Flyefit 3x/week, Sunday run, Friday yoga |
| pref_home | Home Setup | 14 Strand Rd, Nest thermostat, Hue lights |
| pref_language | Language & Format | English, formal, EUR, metric, DD/MM/YYYY |

---

### Event — Live Personal Events (15 nodes)
Timestamped data that changes as Priya's life changes. CRUD operations apply here most.

| ID | Label | Date / Detail |
|----|-------|--------------|
| event_flight | Paris Flight | Ryanair FR2241, 10 Jul 2026 07:45 |
| event_hotel | Paris Hotel | Hotel Le Marais, 10-17 Jul 2026 |
| event_pto | Paris PTO Approved | 10-17 Jul 2026, ref HR2026PTO041 |
| event_rent | Monthly Rent | EUR 1,200 on 1st of month |
| event_fraud | Fraud Dispute Open | FD2026001, EUR 250, resolve 5 Jun 2026 |
| event_medication | Daily Medication Log | Ramipril 5mg at 8am daily |
| event_car | Car Service | Dublin Motors, 15 Jun 2026 10am |
| event_course | Coursera ML Course | DeepLearning.AI, deadline 26 Jul 2026 |
| event_thesis | MSc Thesis Deadline | 12 Sep 2026, NCI Research Practicum |
| event_dentist | Dentist Appointment | Dr Aoife Ryan, 20 Jun 2026 2pm |
| event_gym | Weekly Gym Schedule | Flyefit Tue/Thu/Sat 6:30am |
| event_subscriptions | Monthly Subscriptions | Spotify, Netflix, Coursera — EUR 97.96/mo |
| event_savings | Monthly Savings | EUR 300 on 5th, target EUR 15,000 |
| event_exam | NCI Module Exam | 18 Aug 2026 10am, MSCODP2 |
| event_electricity | Electricity Bill | Electric Ireland, bi-monthly ~EUR 110 |

---

### AccountState — Current Profile Facts (5 nodes)
Static profile data that changes infrequently (monthly/yearly).

| ID | Label | What it holds |
|----|-------|--------------|
| acct_financial | Financial Profile | Balance EUR 3,200, savings EUR 8,400, income EUR 4,500 |
| acct_health | Health Profile | Age 29, Ramipril, VHI Plan B, Dr Nolan |
| acct_work | Work Profile | TechCorp Dublin, Sarah Collins, 25 PTO days |
| acct_education | Education Profile | NCI MSc, Year 2, supervisor Dr Kevin McDaid |
| acct_emergency | Emergency Contacts | Home address, mother, friend, GP, hospital |

---

## How to Add a New Node

1. Open the relevant file in `nodes/` (e.g. `event.json` for a new event)
2. Add a new JSON object following the pattern:
```json
{
  "id": "event_example",
  "type": "Event",
  "label": "Short Display Label",
  "content": "Full descriptive text with all specific facts, dates, amounts, references.",
  "intents": ["check_schedule", "set_reminder"]
}
```
3. Run the regenerate script to update `all_nodes.json`:
```
python emg_data/rebuild.py
```
4. Restart `app.py` — the new node is loaded automatically.

OR use the CRUD API while the app is running:
```
POST /api/crud  {"op": "INSERT", "node_id": "...", "node_type": "...", "content": "...", "label": "..."}
```

---

## Intent → Node Type Routing

The intent classifier routes each query to the most relevant node types.

| Intent | Node Types Searched |
|--------|-------------------|
| check_balance | AccountState → Event |
| report_fraud | Event → FAQ |
| medication_reminder | Event → Preference |
| check_pto | AccountState → Event |
| dietary_advice | Preference → FAQ |
| commute_traffic | Preference → AccountState |
| task_management | Event → AccountState |
| track_fitness | AccountState → Preference |
| check_flight_status | Event → FAQ |
| capabilities | FAQ |

---

*Persona: Priya Sharma | Dublin, Ireland | 2026-05-31*
