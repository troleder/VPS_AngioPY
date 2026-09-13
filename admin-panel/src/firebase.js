import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCCfLa9XjxUf0leHaE4sw4WlzogMOtAFFc",
  authDomain: "registry-coral.firebaseapp.com",
  projectId: "registry-coral",
  storageBucket: "registry-coral.firebasestorage.app",
  messagingSenderId: "250526156796",
  appId: "1:250526156796:web:0eca70f9e8b7490344d479",
  measurementId: "G-4JQE8JC0VB"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

