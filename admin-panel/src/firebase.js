import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAfowkyjJjKDZ6mNyvZJqBk2FoiDMI-iGY",
  authDomain: "coral-registry.firebaseapp.com",
  projectId: "coral-registry",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
