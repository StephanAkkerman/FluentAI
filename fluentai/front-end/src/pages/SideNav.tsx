// src/pages/Header.tsx
import React from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate
import styles from "../styles/sideNavStyles.module.css";

// Import the logo image
import logo from "../icons/Logo (with stroke).png"
import CloseSvg from "../icons/svg/close";

type SideNavProps = {
    show: boolean
    onCloseClick: () => void;
};

const SideNav: React.FC<SideNavProps> = ({ show, onCloseClick }) => {
    const navigate = useNavigate(); // Initialize useNavigate

    const handleMenuClick = (event: React.MouseEvent) => {
        event.stopPropagation(); // Prevents the click from propagating to the parent
    };

    const handleNavigation = (path: string) => {
        navigate(path); // Navigate to the given path
        onCloseClick(); // Close the sidenav after navigation
    };

    return (
        <div className={show ? styles.sideNav : styles.sideNavHidden} onClick={onCloseClick}>
            <div className={show ? styles.menuWrapper : styles.menuWrapperHidden} onClick={handleMenuClick}>
                <div className={styles.header}>
                    <div className={styles.close}>
                        <CloseSvg onClick={onCloseClick} className={styles.icon} />
                    </div>
                    <div className={styles.logo}>
                        <img src={logo} alt="FluentAI Logo" className={styles.appLogo} />
                        <h1 className={styles.appName}>FluentAI</h1>
                    </div>
                </div>

                <div className={styles.menu}>
                    <div className={styles.menuItem}>
                        <h1 onClick={() => handleNavigation("/CardCreation")}>
                            Card Creation
                        </h1>
                        <hr />
                    </div>
                    <div className={styles.menuItem}>
                        <h1 onClick={() => handleNavigation("/CardLibrary")}>
                            Card Library
                        </h1>
                        <hr />
                    </div>
                    <div className={styles.menuItem}>
                        <h1 onClick={() => handleNavigation("/Settings")}>
                            Settings
                        </h1>
                        <hr />
                    </div>
                </div>

            </div>

        </div>

    );
};

export default SideNav;
