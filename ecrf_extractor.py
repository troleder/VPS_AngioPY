"""
eCRF Diagram - Ekstraktor danych angiograficznych pacjenta
===========================================================
Wyciąga dane angiograficzne z formularza eCRF dla podanego numeru randomizacji.

Użycie:
    python ecrf_extractor.py --patient 1701-0030 --login USER --password PASS
    python ecrf_extractor.py --patient 1701-0030  # odczyta LOGIN/PASS ze zmiennych środowiskowych
    python ecrf_extractor.py --patient 1701-0030 --output wyniki.json --headless
"""

import os
import sys
import time
import json
import argparse
import platform
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# ── Konfiguracja ──────────────────────────────────────────────────────────────

BASE  = "https://www.ecrfdiagram.com/eCRF"
STUDY = "4b40cced-3ab5-4f05-91bb-4a2f9bcc9b74"

LOGIN_USER_ID  = "ctl00_Content_ucLogin_txt1_I"
LOGIN_PASS_ID  = "ctl00_Content_ucLogin_txt2_I"
LOGIN_BTN_ID   = "ctl00_Content_ucLogin_cmd1"
FILTER_RAND_ID = "ctl00_Content_ucStudySubjectsPerSite_grdData_DXFREditorcol4_I"


# ── Modele danych ─────────────────────────────────────────────────────────────

@dataclass
class VesselData:
    vessel: str
    segment: str = ""
    culprit: bool = False
    stenosis_pct: Optional[str] = None
    timi_pre: Optional[str] = None
    timi_post: Optional[str] = None
    ffr_performed: bool = False
    ffr_value: Optional[str] = None
    oct_performed: bool = False
    angioplasty_performed: bool = False
    stent_implanted: bool = False
    stent_name: Optional[str] = None
    stent_diameter: Optional[str] = None
    stent_length: Optional[str] = None
    access_site: Optional[str] = None
    guiding_fr: Optional[int] = None
    extension_catheter: Optional[bool] = None

@dataclass
class PatientData:
    randomization_number: str
    site: str
    patient_number: str
    vessels: list = field(default_factory=list)   # list[VesselData]
    sections: dict = field(default_factory=dict)  # section_name → {field: value}


# ── Klasa główna ──────────────────────────────────────────────────────────────

LINUX_CHROME_BINARIES = [
    "/usr/bin/chromium",          # Debian
    "/usr/bin/chromium-browser",  # Ubuntu
    "/snap/bin/chromium",
    "/usr/bin/google-chrome",
]
LINUX_CHROMEDRIVER_BINARIES = [
    "/usr/bin/chromedriver",                  # Debian (chromium-driver)
    "/usr/lib/chromium/chromedriver",         # Debian alt
    "/usr/lib/chromium-browser/chromedriver", # Ubuntu
]


class ECRFExtractor:
    def __init__(self, headless: bool = False):
        opts = Options()
        opts.add_argument("--headless=new")           # modern headless
        opts.add_argument("--window-size=1440,900")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-setuid-sandbox")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--disable-software-rasterizer")
        opts.add_argument("--disable-background-networking")
        opts.add_argument("--disable-default-apps")
        opts.add_argument("--disable-accelerated-2d-canvas")
        opts.add_argument("--no-first-run")
        opts.add_argument("--mute-audio")

        service = None
        if platform.system() == "Linux":
            for binary in LINUX_CHROME_BINARIES:
                if os.path.exists(binary):
                    opts.binary_location = binary
                    break
            for driver_path in LINUX_CHROMEDRIVER_BINARIES:
                if os.path.exists(driver_path):
                    service = Service(driver_path)
                    break

        self.driver = webdriver.Chrome(options=opts, **({"service": service} if service else {}))
        self.driver.implicitly_wait(3)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.driver.quit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wait_dx(self, timeout: int = 20) -> None:
        """Czeka na zakończenie żądań AJAX (DevExpress + jQuery)."""
        for script in [
            "try{return ASPx.IsRequestInProgress()===false}catch(e){return true}",
            "return document.readyState==='complete'",
        ]:
            try:
                WebDriverWait(self.driver, timeout).until(
                    lambda d, s=script: d.execute_script(s)
                )
            except Exception:
                pass
        time.sleep(1.5)

    # ── Logowanie ─────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> None:
        print(f"[1] Logowanie do eCRF...")
        self.driver.get(f"{BASE}/Authenticate.aspx")
        self._wait_dx(15)

        u = WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.ID, LOGIN_USER_ID))
        )
        p = self.driver.find_element(By.ID, LOGIN_PASS_ID)
        u.clear(); u.send_keys(username)
        p.clear(); p.send_keys(password)
        self.driver.find_element(By.ID, LOGIN_BTN_ID).click()

        # Czekaj aż URL zmieni się z Authenticate.aspx (max 30s)
        try:
            WebDriverWait(self.driver, 30).until(
                lambda d: "Authenticate.aspx" not in d.current_url
            )
        except Exception:
            body = self.driver.find_element(By.TAG_NAME, "body").text[:300]
            raise RuntimeError(
                f"Logowanie nie powiodło się. Sprawdź login i hasło.\n{body}"
            )

        print(f"    URL po logowaniu: {self.driver.current_url}")
        print("    Ustawiam kontekst studium...")
        self.driver.get(f"{BASE}/Study.aspx?ID={STUDY}")
        self._wait_dx(25)
        print(f"    URL po study: {self.driver.current_url}")

    # ── Wyszukiwanie pacjenta ─────────────────────────────────────────────────

    def find_patient_guid(self, rand: str, site: str) -> Optional[str]:
        """
        Otwiera BrowseSubjects, filtruje po numerze randomizacji,
        wyciąga GUID pacjenta z onclick="GotoStudySubject('GUID')".
        Zwraca GUID lub None.
        """
        print(f"\n[2] Nawiguję do BrowseSubjects (site={site})...")
        self.driver.get(f"{BASE}/BrowseSubjects.aspx?site={site}")

        print("    Czekam na załadowanie gridu...")
        WebDriverWait(self.driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tr[id*='DXDataRow']"))
        )
        self._wait_dx()

        print(f"    Filtruję po '{rand}'...")
        fld = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, FILTER_RAND_ID))
        )
        fld.clear()
        fld.send_keys(rand)
        fld.send_keys(Keys.RETURN)
        self._wait_dx(25)

        # Wyciągnij GUID z onclick GotoStudySubject
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        for el in soup.find_all(onclick=True):
            m = re.search(r"GotoStudySubject\(['\"]([0-9a-f-]{36})['\"]", el["onclick"])
            if m:
                guid = m.group(1)
                print(f"    Znaleziono GUID: {guid}")
                return guid

        # Fallback: szukaj linku do StudySubject.aspx
        for a in soup.find_all("a", href=True):
            m = re.search(r"StudySubject\.aspx\?ID=([0-9a-f-]{36})", a["href"])
            if m:
                guid = m.group(1)
                print(f"    Znaleziono GUID (link): {guid}")
                return guid

        print(f"    [!] Nie znaleziono GUID dla pacjenta {rand}")
        return None

    # ── Nawigacja po CRF pacjenta ─────────────────────────────────────────────

    def collect_patient_data(self, guid: str) -> dict:
        """
        Otwiera StudySubject.aspx, zbiera linki do formularzy i odwiedza każdy.
        Zwraca słownik: section_name → {field: value}.
        """
        print(f"\n[3] Otwieranie karty pacjenta (GUID={guid})...")
        self.driver.get(f"{BASE}/StudySubject.aspx?ID={guid}")
        self._wait_dx(25)
        print(f"    URL: {self.driver.current_url}")

        all_data: dict = {}
        visited: set = {self.driver.current_url}

        # Dane z głównej strony pacjenta
        main_fields = self._extract_fields()
        if main_fields:
            all_data["overview"] = main_fields
            print(f"    overview: {len(main_fields)} pól")

        # Zbierz linki do sekcji CRF
        crf_links = self._get_crf_links()
        print(f"\n    Sekcje CRF ({len(crf_links)}):")
        for name, url in list(crf_links.items()):
            print(f"      [{name[:60]}]")

        def visit(name: str, url: str) -> None:
            if url in visited:
                return
            visited.add(url)
            print(f"\n    Odwiedzam: {name[:60]}")
            try:
                self.driver.get(url)
                self._wait_dx()
                fields = self._extract_fields()
                if fields:
                    all_data[name] = fields
                    print(f"      {len(fields)} pól")
                    self._print_clinical_highlights(fields)
                # Rekurencja: sub-linki
                sub_links = self._get_crf_links()
                for sname, surl in sub_links.items():
                    if surl not in visited and sname not in crf_links:
                        visit(f"{name} > {sname}", surl)
            except Exception as e:
                print(f"      [!] {e}")

        for name, url in list(crf_links.items()):
            visit(name, url)

        return all_data

    def _get_crf_links(self) -> dict:
        """Wyciąga linki do sekcji CRF z bieżącej strony."""
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        links: dict = {}

        SKIP = {"logoff", "contact", "password", "javascript:", "message",
                "queryresult", "default.aspx", "disclaimer", "authenticate",
                "changepassword", "select study", "browse"}

        CRF_KW = ["procedure", "lesion", "stent", "target", "discharge", "baseline",
                  "imaging", "intervention", "inclusion", "exclusion", "adverse",
                  "event", "deviation", "crf", "form", "follow", "study end",
                  "subject detail", "hidden"]

        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if not href or not text:
                continue
            if any(s in href.lower() or s in text.lower() for s in SKIP):
                continue
            if any(kw in text.lower() or kw in href.lower() for kw in CRF_KW):
                full = href if href.startswith("http") else f"{BASE}/{href.lstrip('/')}"
                links[text] = full

        for el in soup.find_all(onclick=True):
            text = el.get_text(strip=True)
            oc = el.get("onclick", "")
            m = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", oc)
            if m:
                href = m.group(1)
                if not any(s in href.lower() for s in SKIP):
                    if any(kw in text.lower() or kw in href.lower() for kw in CRF_KW):
                        full = href if href.startswith("http") else f"{BASE}/{href.lstrip('/')}"
                        if text:
                            links[text] = full

        return links

    def _extract_fields(self) -> dict:
        """
        Wyciąga wszystkie widoczne pola formularza z bieżącej strony.

        Struktura eCRF Diagram (DevExpress ASP.NET):
          <table class="FormRow">
            <td class="FormRowQuestion"><span>Pytanie</span></td>
            <td class="FormRowInput"><input value="wartość" / radio / select /></td>
          </table>
        """
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        data: dict = {}

        # ── Metoda 1: FormRow (główny schemat eCRF Diagram) ─────────────────
        for row in soup.find_all("tr"):
            q_td = row.find("td", class_="FormRowQuestion")
            i_td = row.find("td", class_="FormRowInput")
            if not q_td or not i_td:
                continue

            k = q_td.get_text(" ", strip=True)
            if not k or len(k) > 150:
                continue

            v = ""

            # Pole tekstowe / numeryczne (obsługujemy brak type, co domyślnie oznacza text, wykluczamy inne typy)
            for inp in i_td.find_all("input"):
                t = inp.get("type", "text").lower()
                if t not in ("hidden", "button", "submit", "checkbox", "radio"):
                    val = inp.get("value", "").strip()
                    if val:
                        v = val
                        break

            # Textarea (pola komentarzy itp.)
            if not v:
                textarea = i_td.find("textarea")
                if textarea:
                    v = textarea.get_text(strip=True)

            # Select (dropdown)
            if not v:
                sel = i_td.find("select")
                if sel:
                    opt = sel.find("option", selected=True)
                    if opt:
                        v = opt.get_text(strip=True)

            # Radio (zaznaczony)
            if not v:
                for radio in i_td.find_all("input", type="radio"):
                    if radio.get("checked") is not None:
                        rid = radio.get("id", "")
                        lbl = soup.find("label", attrs={"for": rid})
                        if lbl:
                            v = lbl.get_text(strip=True)
                        else:
                            parent_lbl = radio.find_parent("label")
                            if parent_lbl:
                                v = parent_lbl.get_text(strip=True)
                            else:
                                for sib in radio.next_siblings:
                                    if isinstance(sib, str):
                                        sib_t = sib.strip()
                                        if sib_t:
                                            v = sib_t
                                            break
                                    elif sib.name in ("span", "label", "td"):
                                        sib_t = sib.get_text(strip=True)
                                        if sib_t:
                                            v = sib_t
                                            break
                        if not v:
                            v = radio.get("value", "")
                        break

            # Checkbox (zaznaczony — zbierz wszystkie)
            if not v:
                vals = []
                for cb in i_td.find_all("input", type="checkbox"):
                    if cb.get("checked") is not None:
                        rid = cb.get("id", "")
                        lbl = soup.find("label", attrs={"for": rid})
                        cb_val = ""
                        if lbl:
                            cb_val = lbl.get_text(strip=True)
                        else:
                            parent_lbl = cb.find_parent("label")
                            if parent_lbl:
                                cb_val = parent_lbl.get_text(strip=True)
                            else:
                                for sib in cb.next_siblings:
                                    if isinstance(sib, str):
                                        sib_t = sib.strip()
                                        if sib_t:
                                            cb_val = sib_t
                                            break
                                    elif sib.name in ("span", "label", "td"):
                                        sib_t = sib.get_text(strip=True)
                                        if sib_t:
                                            cb_val = sib_t
                                            break
                        if not cb_val:
                            cb_val = cb.get("value", "")
                        if cb_val:
                            vals.append(cb_val)
                if vals:
                    v = ", ".join(vals)

            # Final fallback (np. dla pól tylko do odczytu wyrenderowanych jako <span> lub zwykły tekst)
            if not v:
                has_choice_elements = bool(i_td.find_all("input", type=["radio", "checkbox"])) or bool(i_td.find("select"))
                if not has_choice_elements:
                    v = i_td.get_text(" ", strip=True)

            if k and v:
                data[k] = v

        # ── Metoda 2: label[for=] (fallback dla niestandard. elementów) ─────
        for lbl in soup.find_all("label"):
            k = lbl.get_text(strip=True)
            if not k or len(k) > 150:
                continue
            fid = lbl.get("for")
            if not fid:
                continue
            el = soup.find(id=fid)
            if not el:
                continue
            if el.name == "select":
                opt = el.find("option", selected=True)
                v = opt.get_text(strip=True) if opt else ""
            elif el.name == "input":
                t = el.get("type", "text").lower()
                if t in ("checkbox", "radio"):
                    v = "Yes" if el.get("checked") is not None else ""
                else:
                    v = el.get("value", "").strip()
            elif el.name == "textarea":
                v = el.get_text(strip=True)
            else:
                v = el.get_text(strip=True)
            if k and v and k not in data:
                data[k] = v

        # ── Metoda 3: tabele 2-kolumnowe (meta-dane: Created, Modified…) ────
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) == 2:
                    k = cells[0].get_text(" ", strip=True)
                    v = cells[1].get_text(" ", strip=True)
                    if (k and v and 2 < len(k) < 80 and 0 < len(v) < 200
                            and k not in data):
                        data[k] = v

        return data

    @staticmethod
    def _print_clinical_highlights(fields: dict) -> None:
        KW = ["vessel", "artery", "segment", "lesion", "culprit", "ffr", "oct",
              "timi", "stenosis", "stent", "pci", "flow", "lad", "lcx", "rca",
              "coronary", "angio", "ifa", "ira", "diameter", "calcif", "thrombus",
              "occlusion", "balloon", "recanali", "infarct",
              "guiding", "access site", "french", "catheter", "extension catheter"]
        for k, v in fields.items():
            if any(kw in k.lower() or kw in v.lower() for kw in KW):
                print(f"      ★ {k}: {v}")

    # ── Parsowanie naczyń ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_vessels(sections: dict) -> list:
        """Buduje listę VesselData ze słownika sekcji."""
        VESSEL_KW = {
            "LAD": ["lad", " anterior", "left anterior", "gałąź przednia"],
            "LCX": ["lcx", "cx", "circumflex", "okalająca"],
            "RCA": ["rca", "right coronary", "prawa wieńcowa"],
            "OM":  ["om", "obtuse marginal"],
            "D1":  ["d1", "diagonal"],
            "LM":  ["lm", "left main", "pień"],
            "PDA": ["pda", "posterior descending"],
        }
        ANGIO_KW = {
            "culprit":   ["culprit", "ifa", "ira", "infarct-related", "odpowiedzialn"],
            "ffr":       ["ffr", "fractional flow", "pd/pa", "rfr"],
            "oct":       ["oct", "optical coherence"],
            "stenosis":  ["stenosis", "stenoz", "zwężen"],
            "timi_pre":  ["timi pre", "timi flow pre", "baseline timi", "timi przed"],
            "timi_post": ["timi post", "timi flow post", "final timi", "timi po"],
            "stent":     ["stent", "des", "bes", "bms"],
            "pci":       ["pci", "angioplast", "ptca", "intervention"],
            "access":    ["access site"],
            "guiding":   ["guiding"],
            "ext_cath":  ["extension catheter"],
        }

        def identify_vessel(text: str) -> Optional[str]:
            t = text.lower()
            for name, kws in VESSEL_KW.items():
                if any(kw in t for kw in kws):
                    return name
            return None

        vessel_map: dict = {}

        for sec_name, fields in sections.items():
            for fkey, fval in fields.items():
                combined = f"{sec_name} {fkey} {fval}".lower()

                vessel = identify_vessel(combined)
                if not vessel:
                    vessel = identify_vessel(fval)
                if not vessel:
                    vessel = "_unknown"

                if vessel not in vessel_map:
                    vessel_map[vessel] = VesselData(vessel=vessel if vessel != "_unknown" else "?")
                vd = vessel_map[vessel]
                fk = fkey.lower()
                fv = fval.lower()

                # Segment
                if "segment" in fk and not vd.segment:
                    vd.segment = fval

                # Culprit
                if any(kw in fk for kw in ANGIO_KW["culprit"]):
                    if fv in ("yes", "tak", "true", "1", "+"):
                        vd.culprit = True

                # FFR
                if any(kw in fk for kw in ANGIO_KW["ffr"]):
                    vd.ffr_performed = True
                    m = re.search(r"0\.\d+", fval)
                    if m:
                        vd.ffr_value = m.group()

                # OCT
                if any(kw in fk for kw in ANGIO_KW["oct"]):
                    if fv not in ("no", "nie", "false", "0", ""):
                        vd.oct_performed = True

                # Stenoza
                if any(kw in fk for kw in ANGIO_KW["stenosis"]):
                    if fval and not vd.stenosis_pct:
                        vd.stenosis_pct = fval

                # TIMI pre
                if any(kw in fk for kw in ANGIO_KW["timi_pre"]):
                    vd.timi_pre = fval

                # TIMI post
                if any(kw in fk for kw in ANGIO_KW["timi_post"]):
                    vd.timi_post = fval

                # Stent / PCI
                if any(kw in fk for kw in ANGIO_KW["stent"]):
                    vd.stent_implanted = True
                    vd.angioplasty_performed = True
                    if fv not in ("no", "nie", "false", "0", ""):
                        if re.search(r"\d+[x×]\d+", fval):
                            vd.stent_name = fval
                elif any(kw in fk for kw in ANGIO_KW["pci"]):
                    if fv not in ("no", "nie", "false", "0", ""):
                        vd.angioplasty_performed = True

                # Access site (Femoralis / Radialis / Brachialis)
                if any(kw in fk for kw in ANGIO_KW["access"]):
                    if fval and not vd.access_site:
                        vd.access_site = fval

                # Guiding catheter size in French (5 FR / 6 FR / 7 FR / 8 FR)
                if any(kw in fk for kw in ANGIO_KW["guiding"]):
                    m = re.search(r"\b([5-8])\s*[Ff][Rr]\b", fval)
                    if m and not vd.guiding_fr:
                        vd.guiding_fr = int(m.group(1))

                # Extension catheter used
                if any(kw in fk for kw in ANGIO_KW["ext_cath"]):
                    if vd.extension_catheter is None:
                        if fv in ("yes", "tak", "true", "1", "+"):
                            vd.extension_catheter = True
                        elif fv in ("no", "nie", "false", "0", "-"):
                            vd.extension_catheter = False

        real = {k: v for k, v in vessel_map.items() if k != "_unknown"}
        unk = vessel_map.get("_unknown")
        vessels = list(real.values())
        if unk and (unk.stenosis_pct or unk.ffr_performed or unk.oct_performed):
            vessels.append(unk)
        return vessels

    # ── Lista pacjentów (site lub wszyscy) ───────────────────────────────────

    def list_all_patients(self) -> list:
        """
        Zwraca listę numerów randomizacji WSZYSTKICH pacjentów we wszystkich sites.
        """
        return self._list_patients_from_grid(site=None)

    def list_site_patients(self, site: str) -> list:
        """
        Zwraca listę numerów randomizacji wszystkich pacjentów dla danego site.
        Obsługuje paginację gridu DevExpress.
        """
        return self._list_patients_from_grid(site=site)

    def _list_patients_from_grid(self, site) -> list:
        """
        Wewnętrzna metoda — pobiera listę pacjentów z gridu BrowseSubjects.
        site=None → wszyscy pacjenci; site='XXXX' → filtr po site.
        """
        if site:
            print(f"\n[*] Pobieranie listy pacjentów dla site={site}...")
            self.driver.get(f"{BASE}/BrowseSubjects.aspx?site={site}")
        else:
            print(f"\n[*] Pobieranie listy WSZYSTKICH pacjentów...")
            self.driver.get(f"{BASE}/BrowseSubjects.aspx")

        # Poczekaj na załadowanie gridu lub komunikatu braku wyników
        try:
            WebDriverWait(self.driver, 45).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, "tr[id*='DXDataRow']")
                or d.find_elements(By.CSS_SELECTOR, "td.dxgv")
                or d.find_elements(By.CSS_SELECTOR, ".dxgvEmptyDataRow")
            )
        except Exception:
            print("    [!] Timeout oczekiwania na grid — brak danych?")
            return []
        self._wait_dx()

        # Wzorzec pasujący do numerów randomizacji
        if site:
            site_pattern = re.compile(rf'\b{re.escape(site)}-\d{{4}}\b')
        else:
            site_pattern = re.compile(r'\b\d{4}-\d{4}\b')

        patients = []
        page = 0
        MAX_PAGES = 500   # zabezpieczenie na wypadek pętli nieskończonej

        while page < MAX_PAGES:
            page += 1
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            for row in soup.find_all("tr", id=re.compile(r"DXDataRow")):
                text = row.get_text(" ", strip=True)
                matches = site_pattern.findall(text)
                patients.extend(matches)

            # Usuń duplikaty na bieżąco
            patients = list(dict.fromkeys(patients))
            count = len(patients)
            print(f"    Strona {page}: {count} pacjentów łącznie")

            # Sprawdź przycisk "Następna strona" — czy istnieje i nie jest wyłączony
            nxt_btns = self.driver.find_elements(
                By.CSS_SELECTOR,
                "a[title='Next Page'], a[title='Następna strona'], "
                "img[alt='Next'], td[title='Next Page']"
            )
            if not nxt_btns:
                print("    [*] Brak przycisku 'Next Page' — koniec paginacji.")
                break

            nxt = nxt_btns[0]
            nxt_class = (nxt.get_attribute("class") or "").lower()
            # DevExpress używa klas: dxp-disabledButton, disabled, dxp-bi (brak następnej)
            if any(d in nxt_class for d in ("disabled", "dxp-bi", "dxp-disabledbutton")):
                print("    [*] Przycisk 'Next Page' wyłączony — ostatnia strona.")
                break

            # Sprawdź też element nadrzędny (td/span) — często to on ma klasę disabled
            try:
                parent_class = (nxt.find_element(By.XPATH, "..").get_attribute("class") or "").lower()
                if any(d in parent_class for d in ("disabled", "dxp-bi", "dxp-disabledbutton")):
                    print("    [*] Kontener 'Next Page' wyłączony — ostatnia strona.")
                    break
            except Exception:
                pass

            # Kliknij "Następna strona"
            try:
                nxt.click()
                self._wait_dx()
            except Exception as e:
                print(f"    [*] Nie można kliknąć 'Next Page': {e}")
                break

        if page >= MAX_PAGES:
            print(f"    [!] Osiągnięto limit {MAX_PAGES} stron — przerywam.")

        result = list(dict.fromkeys(patients))
        label = f"site {site}" if site else "wszystkich sites"
        print(f"    Znaleziono {len(result)} unikalnych pacjentów ({label})")
        return result

    # ── Główny punkt wejścia ──────────────────────────────────────────────────

    def extract(self, username: str, password: str,
                randomization_number: str) -> PatientData:
        parts = randomization_number.split("-")
        site = parts[0]
        patient_num = parts[1] if len(parts) > 1 else ""

        self.login(username, password)
        guid = self.find_patient_guid(randomization_number, site)
        if not guid:
            raise RuntimeError(f"Nie znaleziono pacjenta {randomization_number}")

        sections = self.collect_patient_data(guid)
        vessels = self._parse_vessels(sections)

        return PatientData(
            randomization_number=randomization_number,
            site=site,
            patient_number=patient_num,
            vessels=vessels,
            sections=sections,
        )


# ── Raport kliniczny ──────────────────────────────────────────────────────────

def print_report(data: PatientData) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  RAPORT ANGIOGRAFICZNY — PACJENT {data.randomization_number}")
    print(f"{sep}")

    ANGIO_KW = ["vessel", "artery", "segment", "lesion", "culprit", "ffr", "oct",
                "timi", "stenosis", "stent", "pci", "flow", "lad", "lcx", "rca",
                "coronary", "angio", "ifa", "ira", "diameter", "calcif", "thrombus",
                "occlusion", "balloon", "bifurc", "recanali", "revas", "angina",
                "infarct", "disease", "target", "imaging", "measurement"]

    found_any = False
    for sec, fields in data.sections.items():
        rel = {k: v for k, v in fields.items()
               if any(kw in k.lower() or kw in v.lower() for kw in ANGIO_KW)}
        if rel:
            found_any = True
            print(f"\n  ┌─ {sec}")
            for k, v in rel.items():
                print(f"  │  {k}: {v}")
            print(f"  └─")

    if not found_any:
        print("\n  [Brak pól klinicznych — pełna lista zebranych danych]")
        for sec, fields in data.sections.items():
            if fields:
                print(f"\n  [{sec}]")
                for k, v in list(fields.items())[:30]:
                    print(f"    {k}: {v}")

    print()

    if data.vessels:
        print(f"  {'─'*65}")
        print(f"  NACZYNIA (parsowane automatycznie)")
        print(f"  {'─'*65}")
        for v in data.vessels:
            print(f"\n  Naczynie   : {v.vessel}")
            print(f"  Segment    : {v.segment or '—'}")
            print(f"  Culprit    : {'TAK ★' if v.culprit else 'Nie'}")
            print(f"  Stenoza    : {v.stenosis_pct or '—'}")
            print(f"  TIMI pre   : {v.timi_pre or '—'}")
            print(f"  TIMI post  : {v.timi_post or '—'}")
            if v.ffr_performed:
                print(f"  FFR        : TAK  →  {v.ffr_value or 'brak wartości'}")
            else:
                print(f"  FFR        : Nie wykonano")
            print(f"  OCT        : {'TAK' if v.oct_performed else 'Nie wykonano'}")
            if v.access_site:
                print(f"  Dostęp        : {v.access_site}")
            if v.guiding_fr:
                print(f"  Cewnik prowadz: {v.guiding_fr} FR")
            if v.extension_catheter is not None:
                print(f"  Ext. cewnik   : {'TAK' if v.extension_catheter else 'Nie'}")
            if v.angioplasty_performed:
                stent_info = ""
                if v.stent_implanted:
                    parts = []
                    if v.stent_name:     parts.append(v.stent_name)
                    if v.stent_diameter: parts.append(f"⌀{v.stent_diameter}")
                    if v.stent_length:   parts.append(f"dł. {v.stent_length}")
                    stent_info = f"  (stent: {', '.join(parts)})" if parts else "  (stent)"
                print(f"  Angioplastyka : TAK{stent_info}")
            else:
                print(f"  Angioplastyka : Nie")
        print()


# ── Czysty eksport JSON ───────────────────────────────────────────────────────

def _fv(fields: dict, *keys: str) -> str:
    """Zwraca pierwszą wartość z fields której klucz zawiera któryś z keys."""
    for k, v in fields.items():
        if any(key.lower() in k.lower() for key in keys):
            if v and v.strip():
                return v.strip()
    return ""


def _first(sections: dict, *keys: str) -> str:
    """Szuka wartości po keys we wszystkich sekcjach."""
    for fields in sections.values():
        val = _fv(fields, *keys)
        if val:
            return val
    return ""


def _to_bool(val: str):
    if not val:
        return None
    v = val.strip().lower()
    if v in {"yes", "tak", "true", "1", "+", "checked"}:
        return True
    if v in {"no", "nie", "false", "0", "-"}:
        return False
    return None


def _to_float(val: str):
    if not val:
        return None
    try:
        return float(val.replace(",", ".").strip())
    except ValueError:
        return None


def _to_int(val: str):
    if not val:
        return None
    m = re.search(r"\d+", val)
    return int(m.group()) if m else None


TIMI_MAP = {"0": 0, "I": 1, "II": 2, "III": 3, "1": 1, "2": 2, "3": 3}


def build_clean_json(data: PatientData) -> dict:
    """
    Przekształca surowy PatientData w czysty, ustrukturyzowany dict gotowy do exportu JSON.

    Struktura wyjściowa:
    {
      "schema_version": "1.0",
      "patient": { ... },
      "vessels": [
        {
          "segment": "01= RCA proximal",
          "culprit": true,
          "stenosis_pct": 95,
          "timi_pre": 1,
          "physiology_done": false,
          "pd_pa": null,
          "rfr": null,
          "ffr_adenosine": null,
          "oct_pre": false,
          "pci_performed": true,
          "stents": [ {"type": "XIENCE DES", "diameter_mm": 3.0, "length_mm": 33} ],
          "timi_post": 3,
          "residual_stenosis": "< 30%",
          "pci_successful": true,
          ...
        }
      ]
    }
    """
    sections = data.sections

    # ── Dane pacjenta ────────────────────────────────────────────────────────
    arm = _first(sections, "Patient is randomized to", "randomized to")
    if not arm:
        arm = _first(sections, "Experimental arm (FFR", "Comparative arm (FFR")

    patient = {
        "randomization_number": data.randomization_number,
        "site":           data.site,
        "patient_number": data.patient_number,
        "arm":            arm or None,
        "sex":            _first(sections, "Sex", "Male", "Female") or None,
        "indication":     _first(sections, "Indication", "ACS", "IAP", "STEMI", "NSTEMI") or None,
        "lvef":           _first(sections, "LVEF", "ejection fraction") or None,
        "nyha":           _first(sections, "NYHA") or None,
        "diabetes":       _to_bool(_first(sections, "Diabetes")),
        "smoking":        _first(sections, "Smoking", "Current smoker", "Former smoker") or None,
        "hypertension":   _to_bool(_first(sections, "Hypertension")),
        "previous_pci":   _to_bool(_first(sections, "Previous PCI")),
        "date_angio":     _first(sections, "Date coronary angiography", "Date procedure") or None,
    }

    # ── Klasyfikacja sekcji ──────────────────────────────────────────────────
    imaging_sections      = []   # mają TIMI pre
    intervention_sections = []   # mają "PCI procedure performed"
    stent_sections        = []   # mają "Type of stent"

    for fields in sections.values():
        keys_lower = {k.lower() for k in fields}
        if any("type of stent" in k for k in keys_lower):
            stent_sections.append(fields)
        elif any("initial timi flow" in k for k in keys_lower):
            imaging_sections.append(fields)
        elif any("pci procedure performed" in k for k in keys_lower):
            intervention_sections.append(fields)

    def get_segment(fields: dict) -> str:
        return _fv(fields, "Target Lesion segment").strip()

    imaging_by_seg  = {get_segment(f): f for f in imaging_sections  if get_segment(f)}
    interv_by_seg   = {get_segment(f): f for f in intervention_sections if get_segment(f)}

    all_segments = sorted(
        set(imaging_by_seg) | set(interv_by_seg),
        key=lambda s: s[:2]
    )

    # ── Naczynia ─────────────────────────────────────────────────────────────
    vessels_out = []
    stent_idx = 0

    for seg in all_segments:
        img = imaging_by_seg.get(seg, {})
        inv = interv_by_seg.get(seg, {})

        timi_pre_raw  = _fv(img, "Initial TIMI flow")
        timi_post_raw = _fv(inv, "TIMI post")
        n_stents      = _to_int(_fv(inv, "Number of stents"))

        # Stenty dla tego naczynia
        vessel_stents = []
        if n_stents:
            for _ in range(n_stents):
                if stent_idx >= len(stent_sections):
                    break
                sf = stent_sections[stent_idx]
                vessel_stents.append({
                    "type":        _fv(sf, "Type of stent", "Stent form") or None,
                    "diameter_mm": _to_float(_fv(sf, "Diameter")),
                    "length_mm":   _to_int(_fv(sf, "Length")),
                })
                stent_idx += 1

        vessel = {
            "segment":              seg,
            "culprit":              _to_bool(_fv(img, "Culprit lesion")),
            "stenosis_pct":         _to_int(_fv(img, "Visually estimated diameter stenosis")),
            "timi_pre":             TIMI_MAP.get(timi_pre_raw),
            "physiology_done":      _to_bool(_fv(img, "Physiology measurements performed")),
            "pd_pa":                _to_float(_fv(img, "Pd/Pa")),
            "rfr":                  _to_float(_fv(img, "RFR")),
            "ffr_adenosine":        _to_float(_fv(img, "FFR adenosine")),
            # ── OCT pre-PCI (pola 6–6.12) ──────────────────────────────────
            "oct_pre":                  _to_bool(_fv(img, "OCT performed")),
            "oct_lesion_prep":          _to_bool(_fv(img, "Lesion preparation prior to OCT")),
            "oct_catheter":             _fv(img, "Catheter", "Optis", "Opstar") or None,
            "oct_pullback":             _fv(img, "Pullback length", "54 mm", "75 mm") or None,
            "oct_tcfa":                 _to_bool(_fv(img, "TCFA")),
            "oct_plaque_rupture":       _to_bool(_fv(img, "Plaque rupture")),
            "oct_plaque_erosion":       _to_bool(_fv(img, "Plaque erosion")),
            "oct_pct_lumen_stenosis":   _to_int(_fv(img, "Percent lumen area stenosis")),
            "oct_mla_mm2":              _to_float(_fv(img, "Minimal lumen area")),
            "oct_lesion_length_mm":     _to_float(_fv(img, "Lesion length")),
            "oct_proximal_diam_mm":     _to_float(_fv(img, "Lesion proximal mean diameter")),
            "oct_distal_diam_mm":       _to_float(_fv(img, "Lesion distal mean diameter")),
            # ── Hemodynamika ─────────────────────────────────────────────────
            "pci_performed":        _to_bool(_fv(inv, "PCI procedure performed")),
            "access_site":          _fv(inv, "Access site") or None,
            "guiding_fr":           int(re.search(r"\b([5-8])\s*[Ff][Rr]\b", _fv(inv, "Guiding")).group(1)) if re.search(r"\b([5-8])\s*[Ff][Rr]\b", _fv(inv, "Guiding")) else None,
            "extension_catheter":   _to_bool(_fv(inv, "Extension catheter")),
            "bifurcation":          _to_bool(_fv(inv, "Bifurcation")),
            "calcification":        _fv(inv, "Calcification") or None,
            "predilatation":        _to_bool(_fv(inv, "Balloon predilatation")),
            "predil_balloon_mm":    _to_float(_fv(inv, "Predilation-balloon size")),
            "predil_pressure_atm":  _to_int(_fv(inv, "Highest pressure largest predilation")),
            "stent_placed":         _to_bool(_fv(inv, "Stent placement")),
            "n_stents":             n_stents,
            "stents":               vessel_stents,
            "postdil_balloon_mm":   _to_float(_fv(inv, "Balloon size (max. diameter)")),
            "postdil_pressure_atm": _to_int(_fv(inv, "Highest pressure largest balloon")),
            "timi_post":            TIMI_MAP.get(timi_post_raw),
            "residual_stenosis":    _fv(inv, "Visual Ds post", "residual stenosis") or None,
            "pci_successful":       _to_bool(_fv(inv, "PCI successful")),
            "oct_post":             _to_bool(_fv(inv, "OCT performed post stenting")),
        }
        vessels_out.append(vessel)

    return {
        "schema_version": "1.0",
        "patient":        patient,
        "vessels":        vessels_out,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wyciąga dane angiograficzne z eCRF Diagram"
    )
    parser.add_argument("--patient", "-p", required=True,
                        help="Numer randomizacji: XXXX-AAAA  np. 1701-0030")
    parser.add_argument("--login", "-u",
                        default=os.environ.get("ECRF_LOGIN", ""),
                        help="Login (lub zmienna ECRF_LOGIN)")
    parser.add_argument("--password", "-pw",
                        default=os.environ.get("ECRF_PASSWORD", ""),
                        help="Hasło (lub zmienna ECRF_PASSWORD)")
    parser.add_argument("--headless", action="store_true",
                        help="Chrome bez okna")
    parser.add_argument("--output", "-o", default="",
                        help="Zapisz wyniki JSON do pliku")
    args = parser.parse_args()

    if not args.login or not args.password:
        print("[!] Podaj login i hasło (--login / --password lub ECRF_LOGIN / ECRF_PASSWORD)")
        sys.exit(1)

    if not re.match(r"^\d{4}-\d{4}$", args.patient):
        print(f"[!] Nieprawidłowy format: '{args.patient}'. Oczekiwany: XXXX-AAAA  np. 1701-0030")
        sys.exit(1)

    with ECRFExtractor(headless=args.headless) as ex:
        data = ex.extract(args.login, args.password, args.patient)

    print_report(data)

    # Czysty ustrukturyzowany JSON (główny output)
    clean = build_clean_json(data)
    out = args.output or f"result_{args.patient}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"  Dane zapisane: {out}")
    print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
