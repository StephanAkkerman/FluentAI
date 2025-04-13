"use client"
import WhatSection from "./landing-page/WhatSection"
import WhySection from "./landing-page/WhySection";
import HowSection from "./landing-page/HowSection";
import HeroSection from "./landing-page/HeroSection";
import AIPage from "./landing-page/AISection";
import FAQSection from "./landing-page/FAQSection";
import FooterSection from "./landing-page/FooterSection";
import React from "react";

const LandingPage = () => {
  return (
    <>
      <div className="min-h-screen flex flex-col w-full">
        <section id="home">
          <HeroSection />
        </section>


        <section id="what" className="duration-300 transition-all -translate-y-[5%]">
          <WhatSection />
        </section>

        <section id="why" className="duration-300 transition-all -translate-y-[5%]">
          <WhySection />
        </section>

        {/* 2. HOW IT WORKS SECTION */}
        < section id="how" className="py-20" >
          <HowSection />
        </section >

        {/* 3. FEATURES & BENEFITS SECTION */}
        < section id="features" className="py-20" >
          <AIPage />
        </section >

        {/* 5. FAQ SECTION */}
        < section id="faq" >
          <FAQSection />
        </section >

        {/* 6. FOOTER */}
        < section >
          <FooterSection />
        </section >
      </div >
    </>
  );
};

export default LandingPage;
