import React from 'react';
import '../styles/components/ButtonStyles.css';

type ButtonProps = {
    onClick: any | (() => void); // ()
    text: string;
    type?: 'button' | 'submit' | 'reset';
    style?: {
        cancel?: boolean;
    }
};

const Button: React.FC<ButtonProps> = ({ onClick, text, type, style }) => {
    return (
        <button className={`button${style?.cancel ? '-cancel' : ''}`} onClick={onClick} type={type} >
            {text}
        </button>
    );
};

export default Button;