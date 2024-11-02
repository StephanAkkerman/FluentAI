import React from 'react';
import styles from '../styles/LoadingPageStyles';

const LoadingPage: React.FC<{ isLoading: boolean }> = ({ isLoading }) => {
    const wrapperClass = isLoading ? 'load-wrapper' : 'load-wrapper load-wrapper-completed';

    return (
        <div className={wrapperClass} style={styles.loadWrapper}>
            <div className="loading-text" style={styles.loadingText}>
                Loading...
            </div>
        </div>
    );
};

export default LoadingPage;
